import streamlit as st
from PyPDF2 import PdfReader
import requests
import json
import re # Import the regular expression module
import os

# Import the email sending utility function
from email_utils import send_rejection_email

# --- Configuration for watsonx.ai ---
# IMPORTANT: Replace with your actual API key and Project ID
# Keep these hardcoded as per your preference, but ensure they are correct.
YOUR_IBM_CLOUD_API_KEY = "0FWklQEoW7InnT8Ei6TFB8TdjnzZ6yO6CDDMT-yJy3Fp"
YOUR_WATSONX_PROJECT_ID = "ae83a9cf-aab9-411a-80a1-964fa621ddd0"

# Endpoint URLs for watsonx.ai
# Using v1 based on your provided curl/python details
WATSONX_AI_GENERATION_ENDPOINT = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Watson Data Platform API for Asset Management (important for fetching prompts)
WATSONX_DATA_PLATFORM_API_BASE = "https://api.dataplatform.cloud.ibm.com"
WATSONX_DATA_PLATFORM_API_VERSION = "2024-07-29" # IMPORTANT: Use a current valid version!

# --- Constants for Prompt Names ---
PROMPT_TEMPLATE_NAME_ATS = "recruit_evaluator"
PROMPT_TEMPLATE_NAME_FEEDBACK = "feedback_generator" # Name for feedback generation prompt

# --- watsonx.ai API Functions ---

@st.cache_data(ttl=300) # Cache token for 5 minutes
def _get_iam_token_from_api(api_key):
    """Fetches an IAM token directly from the API and caches it."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + api_key
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching IAM token: {e}")
        return None

def get_iam_token(api_key):
    """
    Manages fetching and storing the IAM token in st.session_state.
    Only calls the API if the token is not in session state or is None.
    """
    if "IAM_TOKEN" not in st.session_state or st.session_state.IAM_TOKEN is None:
        st.session_state.IAM_TOKEN = _get_iam_token_from_api(api_key)
    return st.session_state.IAM_TOKEN


@st.cache_data(ttl=3600) # Cache prompt content for 1 hour
def get_prompt_template_content_by_name(api_key, project_id, template_name):
    """
    Fetches the content of a prompt template from watsonx.ai assets by its name.
    Handles content stored as COS attachments or direct input_text/instruction.
    """
    token = st.session_state.get("IAM_TOKEN")
    if not token:
        st.error("IAM Token not available in session state to fetch prompt templates.")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Step 1: Search for the prompt template to get its asset_id
    search_url = f"{WATSONX_DATA_PLATFORM_API_BASE}/v2/asset_types/wx_prompt/search"
    search_params = {
        "version": WATSONX_DATA_PLATFORM_API_VERSION,
        "project_id": project_id
    }
    search_payload = {
        "query": f"asset.name:{template_name}"
    }

    try:
        st.info(f"Searching for prompt template '{template_name}'...")
        search_response = requests.post(search_url, headers=headers, params=search_params, data=json.dumps(search_payload))
        search_response.raise_for_status()
        search_results = search_response.json()

        assets = search_results.get("results", [])
        if not assets:
            st.error(f"Prompt template '{template_name}' not found in project '{project_id}'.")
            return None

        prompt_asset_id = assets[0]["metadata"]["asset_id"]
        st.success(f"Found prompt template '{template_name}' with ID: {prompt_asset_id}")

        # Step 2: Fetch the actual metadata of the prompt template asset
        get_content_url = f"{WATSONX_DATA_PLATFORM_API_BASE}/v2/assets/{prompt_asset_id}"
        get_content_params = {
            "version": WATSONX_DATA_PLATFORM_API_VERSION,
            "project_id": project_id
        }

        st.info(f"Fetching metadata for prompt template '{template_name}'...")
        content_response = requests.get(get_content_url, headers=headers, params=get_content_params)
        content_response.raise_for_status()
        asset_metadata = content_response.json()

        prompt_content = None

        # Try to get content from 'input_text' or 'instruction' fields
        prompt_content = asset_metadata.get("entity", {}).get("prompt", {}).get("input_text")
        if not prompt_content:
            prompt_content = asset_metadata.get("entity", {}).get("prompt", {}).get("instruction")

        # If still no content, check COS attachments
        if not prompt_content:
            attachments = asset_metadata.get("attachments")
            if attachments and len(attachments) > 0:
                attachment = attachments[0]
                if attachment.get("mime") == "application/json" and "handle" in attachment:
                    bucket = attachment["handle"].get("bucket")
                    object_key = attachment["handle"].get("key")
                    location = attachment["handle"].get("location")

                    if bucket and object_key and location:
                        cos_endpoint = f"s3.{location}.cloud-object-storage.appdomain.cloud"
                        cos_url = f"https://{bucket}.{cos_endpoint}/{object_key}"

                        st.info(f"Attempting to fetch prompt content from Cloud Object Storage (COS): {cos_url}")

                        try:
                            cos_headers = {"Authorization": f"Bearer {token}"}
                            cos_response = requests.get(cos_url, headers=cos_headers)
                            cos_response.raise_for_status()
                            cos_content_json = cos_response.json()

                            # Assuming prompt content is in cos_content_json['input'][0][0] based on previous findings
                            if "input" in cos_content_json and \
                               isinstance(cos_content_json["input"], list) and \
                               len(cos_content_json["input"]) > 0 and \
                               isinstance(cos_content_json["input"][0], list) and \
                               len(cos_content_json["input"][0]) > 0:
                                prompt_content = cos_content_json["input"][0][0]
                            else:
                                st.warning(f"Could not find prompt content in COS JSON for '{template_name}' in 'input[0][0]'.")

                            if prompt_content:
                                st.success("Successfully fetched prompt content from Cloud Object Storage.")
                            else:
                                st.warning(f"Could not find prompt content in COS JSON for '{template_name}' (expected in 'input[0][0]'.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error fetching prompt content from COS for '{template_name}': {e}. Status: {cos_response.status_code if 'cos_response' in locals() else 'N/A'}")
                        except json.JSONDecodeError as e:
                            st.error(f"Error decoding COS JSON content for prompt '{template_name}': {e}. Check for malformed JSON.")

        if prompt_content:
            st.success(f"Successfully extracted prompt content for '{template_name}'.")
            return prompt_content
        else:
            st.error(f"Could not find the actual prompt text content within the asset details or COS object for '{template_name}'.")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching prompt template from watsonx.ai asset catalog: {e}")
        if 'search_response' in locals():
             st.error(f"Search API troubleshooting: Status {search_response.status_code}, Response: {search_response.text}")
        if 'content_response' in locals():
             st.error(f"Asset Metadata API troubleshooting: Status {content_response.status_code}, Response: {content_response.text}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding API response for prompt template metadata: {e}. Check the raw response above for malformed JSON.")
        return None


def call_watsonx_ai_granite(api_key, project_id, prompt_text, prompt_name, max_tokens=250, temperature=0.1):
    """
    Calls the watsonx.ai Granite model with the given prompt.
    Adjusted to allow different max_tokens/temperature for different prompt types.
    """
    token = st.session_state.get("IAM_TOKEN")
    if not token:
        return {"error": "Error: IAM token not available for model call."}

    url = WATSONX_AI_GENERATION_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Set parameters based on the prompt name
    # Ensure consistency in parameter names if different from defaults for clarity
    if prompt_name == PROMPT_TEMPLATE_NAME_ATS:
        model_params = {
            "decoding_method": "sample",
            "max_new_tokens": max_tokens, # Using max_tokens passed to function
            "min_new_tokens": 10,
            "random_seed": None,
            "stop_sequences": ["<<END>>"],
            "temperature": temperature, # Using temperature passed to function
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        }
    elif prompt_name == PROMPT_TEMPLATE_NAME_FEEDBACK:
        model_params = {
            "decoding_method": "sample",
            "max_new_tokens": max_tokens, # Using max_tokens passed to function
            "min_new_tokens": 10,
            "random_seed": None,
            "stop_sequences": ["###END_EMAIL###"], # Updated stop sequence for feedback
            "temperature": temperature, # Using temperature passed to function
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        }
    else: # Default parameters if prompt_name is not specifically handled
        model_params = {
            "decoding_method": "sample",
            "max_new_tokens": max_tokens,
            "min_new_tokens": 10,
            "random_seed": None,
            "stop_sequences": ["###END_EMAIL###"],
            "temperature": temperature,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        }

    payload = {
        "model_id": "ibm/granite-3-3-8b-instruct", # Your chosen model ID
        "input": prompt_text,
        "parameters": model_params,
        "project_id": project_id
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error calling watsonx.ai: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error calling watsonx.ai: {e}")
        return {"error": f"Request Error: {e}"}


# --- Function to parse ATS evaluation output ---
def parse_ats_evaluation(evaluation_output: str):
    """
    Parses the ATS evaluation output from the Granite model.
    Returns the score (int) and a list of rejection reasons (list of strings),
    ensuring each reason is consistently formatted with a leading hyphen.
    """
    ats_score = 0
    rejection_reasons = []

    # Extract score
    score_match = re.search(r"Overall ATS Match Score: (\d+)%", evaluation_output)
    if score_match:
        ats_score = int(score_match.group(1))

    # Check for PASSED status (implies high score based on prompt rules)
    if "PASSED:" in evaluation_output:
        # No rejection reasons if PASSED
        pass
    else:
        # Extract rejection reasons if not passed. Using re.DOTALL to match across newlines.
        reasons_section_match = re.search(r"Rejection Reasons:\s*\n(- .*?)(?:\n<<END>>|\n\n|$)", evaluation_output, re.DOTALL)
        if reasons_section_match:
            reasons_text = reasons_section_match.group(1).strip()
            # Split by newline, then strip and ensure each reason starts with a hyphen
            raw_reasons = [line.strip() for line in reasons_text.split('\n') if line.strip()]
            for reason in raw_reasons:
                if not reason.startswith('- '):
                    rejection_reasons.append(f"- {reason}") # Add hyphen if missing
                else:
                    rejection_reasons.append(reason) # Keep as is if already has hyphen

    return ats_score, rejection_reasons


# --- PDF Text Extraction Function ---
def extract_text_from_pdf(pdf_file):
    text = ''
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ''
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        text = ""
    return text

# --- NEW: Function to extract email from text ---
def extract_email_from_text(text):
    """
    Extracts the first valid email address found in the given text.
    """
    # This regex attempts to find standard email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    if match:
        return match.group(0)
    return None


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 GraniteHire: Smart Feedback for Resume Optimization & ATS Success")
st.write("Upload a Job Description and multiple Candidate Resumes to get match scores and feedback.")

# Display API Key and Project ID status
if YOUR_IBM_CLOUD_API_KEY == "YOUR_IBM_CLOUD_API_KEY" or YOUR_WATSONX_PROJECT_ID == "YOUR_WATSONX_PROJECT_ID":
    st.error("Please update 'YOUR_IBM_CLOUD_API_KEY' and 'YOUR_WATSONX_PROJECT_ID' in app.py with your actual credentials.")
else:
    # Attempt to get IAM Token immediately using the hardcoded key
    # This call populates st.session_state.IAM_TOKEN
    get_iam_token(YOUR_IBM_CLOUD_API_KEY)

    if st.session_state.get("IAM_TOKEN"):
        st.success("watsonx.ai API credentials detected. IAM Token obtained. Ready to proceed!")
    else:
        st.error("Failed to obtain IAM Token. Please check your API Key and Project ID in app.py.")

# Get prompt template content dynamically for both ATS and Feedback
ats_prompt_template_content = None
feedback_prompt_template_content = None

# Use an expander for prompt loading status messages
with st.expander("Show Prompt Template Loading Status"):
    if st.session_state.get("IAM_TOKEN"):
        # Fetch ATS prompt
        st.info(f"Attempting to fetch prompt template '{PROMPT_TEMPLATE_NAME_ATS}' from watsonx.ai assets.")
        ats_prompt_template_content = get_prompt_template_content_by_name(
            YOUR_IBM_CLOUD_API_KEY, YOUR_WATSONX_PROJECT_ID, PROMPT_TEMPLATE_NAME_ATS
        )
        if ats_prompt_template_content:
            st.success(f"Successfully loaded prompt template '{PROMPT_TEMPLATE_NAME_ATS}'.")
        else:
            st.error("Failed to load ATS prompt template from watsonx.ai. Please check logs and ensure the template name is correct and accessible.")

        # Fetch Feedback prompt
        st.info(f"Attempting to fetch prompt template '{PROMPT_TEMPLATE_NAME_FEEDBACK}' from watsonx.ai assets.")
        feedback_prompt_template_content = get_prompt_template_content_by_name(
            YOUR_IBM_CLOUD_API_KEY, YOUR_WATSONX_PROJECT_ID, PROMPT_TEMPLATE_NAME_FEEDBACK
        )
        if feedback_prompt_template_content:
            st.success(f"Successfully loaded prompt template '{PROMPT_TEMPLATE_NAME_FEEDBACK}'.")
        else:
            st.error("Failed to load feedback prompt template from watsonx.ai. Please check logs and ensure the template name is correct and accessible.")
    else:
        st.warning("Cannot load prompt templates without a valid IAM token. Please provide correct API Key and Project ID.")

# A more concise status message outside the expander, for quick user feedback
if ats_prompt_template_content and feedback_prompt_template_content:
    st.success("All prompt templates loaded successfully and the application is ready!")
elif not st.session_state.get("IAM_TOKEN"):
    st.warning("Please ensure your watsonx.ai API Key and Project ID are correctly configured to load prompts.")
else:
    st.warning("Some prompt templates failed to load. See 'Show Prompt Template Loading Status' for details.")


# File Uploads
jd_file = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"], key="jd_uploader")
resume_files = st.file_uploader("Upload Candidate Resume(s) (TXT or PDF)", type=["txt", "pdf"], accept_multiple_files=True, key="resume_uploader")

jd_content = ""
if jd_file:
    with st.spinner("Processing Job Description..."):
        if jd_file.type == "application/pdf":
            jd_content = extract_text_from_pdf(jd_file)
        else:
            jd_content = jd_file.read().decode("utf-8")
        st.success("Job Description Loaded!")
        st.subheader("Job Description Preview (First 500 chars):")
        st.text(jd_content[:500] + "...")


# --- Evaluate Match Button ---
if st.button("Evaluate Match"):
    # --- Pre-evaluation checks ---
    if not st.session_state.get("IAM_TOKEN"):
        st.error("Cannot evaluate match without a valid IAM token. Please configure your API Key and Project ID.")
        st.stop() # Stop execution if no token
    if not ats_prompt_template_content:
        st.error("Cannot evaluate match. ATS prompt template could not be loaded or is invalid.")
        st.stop() # Stop execution if ATS prompt is missing
    if not jd_content:
        st.warning("Please upload a Job Description.")
        st.stop() # Stop execution if no JD
    if not resume_files:
        st.warning("Please upload at least one Candidate Resume.")
        st.stop() # Stop execution if no resumes
    if not feedback_prompt_template_content: # Ensure feedback prompt is loaded for potential use
        st.error("Feedback prompt template could not be loaded. Feedback generation might be unavailable.")
        # Do not stop here, ATS evaluation can still proceed, but user is warned

    st.info("Sending data to watsonx.ai for evaluation...")

    # Loop through each uploaded resume file
    for i, resume_file in enumerate(resume_files):
        st.subheader(f"🔍 Evaluating Resume {i+1}: {resume_file.name}")
        resume_content = ""
        with st.spinner(f"Processing {resume_file.name}..."):
            if resume_file.type == "application/pdf":
                resume_content = extract_text_from_pdf(resume_file)
            else:
                resume_content = resume_file.read().decode("utf-8")

            if resume_content:
                # Use the dynamically fetched ATS prompt template
                try:
                    full_ats_prompt = ats_prompt_template_content.format(
                        resume_text=resume_content, jd_text=jd_content
                    )
                except KeyError as e:
                    st.error(f"Error: ATS prompt template from watsonx.ai does not contain expected placeholders. Missing '{{{e}}}'. Please ensure your template has {{resume_text}} and {{jd_text}}.")
                    continue # Skip to next resume if template is malformed

                with st.spinner(f"Getting match score for {resume_file.name} from watsonx.ai..."):
                    ats_response_json = call_watsonx_ai_granite(
                        YOUR_IBM_CLOUD_API_KEY,
                        YOUR_WATSONX_PROJECT_ID,
                        full_ats_prompt,
                        PROMPT_TEMPLATE_NAME_ATS, # Pass the prompt name for parameter selection
                        max_tokens=250, # Specific max_tokens for ATS
                        temperature=0.1 # Specific temperature for ATS
                    )

                    if ats_response_json and "results" in ats_response_json and ats_response_json["results"]:
                        ai_response_text = ats_response_json["results"][0]["generated_text"]

                        st.subheader("Raw ATS Evaluation Result from Model:")
                        st.markdown(f"```\n{ai_response_text}\n```")

                        # --- Parse the evaluation output and apply conditional logic ---
                        ats_score, rejection_reasons = parse_ats_evaluation(ai_response_text)

                        st.write(f"**Parsed ATS Match Score:** {ats_score}%")

                        if ats_score >= 80: # Updated to 80% as per your instruction
                            st.success("🎉 **Resume Accepted!** This candidate is a strong match based on core requirements.")
                        else:
                            st.error("❌ **Resume Not Accepted.** Candidate did not meet all core requirements.")
                            st.subheader("Identified Rejection Reasons:")
                            if rejection_reasons:
                                # This loop will now correctly display bullet points
                                for reason in rejection_reasons:
                                    st.write(reason)
                            else:
                                st.write("No specific rejection reasons identified by the model, or unable to parse them.")

                            st.info("Based on the low score, the system will now generate personalized feedback.")

                            # --- Generate Personalized Feedback Email ---
                            if feedback_prompt_template_content and rejection_reasons:
                                # The rejection_reasons list now contains reasons pre-formatted with hyphens.
                                # Join them with newlines for the prompt input.
                                formatted_rejection_reasons = "\n".join(rejection_reasons)

                                try:
                                    # Replace the placeholder in the feedback prompt
                                    full_feedback_prompt = feedback_prompt_template_content.format(
                                        rejection_reasons_list=formatted_rejection_reasons
                                    )
                                except KeyError as e:
                                    st.error(f"Error: Feedback prompt template from watsonx.ai does not contain expected placeholder '{{{e}}}'. Please ensure your template has {{rejection_reasons_list}}.")
                                    continue # Skip to next resume if template is malformed

                                st.info("Generating personalized feedback email...")
                                with st.spinner("Calling watsonx.ai for feedback generation..."):
                                    # Use max_tokens and temperature specific for feedback
                                    feedback_response_json = call_watsonx_ai_granite(
                                        YOUR_IBM_CLOUD_API_KEY,
                                        YOUR_WATSONX_PROJECT_ID,
                                        full_feedback_prompt,
                                        PROMPT_TEMPLATE_NAME_FEEDBACK,
                                        max_tokens=500, # Sufficient max tokens for complete email (as per your request)
                                        temperature=0.1 # As per your request
                                    )

                                if feedback_response_json and "results" in feedback_response_json and feedback_response_json["results"]:
                                    raw_feedback_body = feedback_response_json["results"][0]["generated_text"].replace("###END_EMAIL###", "").strip()

                                    # --- NEW: Construct the full email with standard intro/outro ---
                                    # This is where we ensure only one intro and outro
                                    full_email_parts = [
                                        "Dear Candidate,",
                                        "", # Add a blank line after salutation
                                        "Thank you for your interest in the Data Scientist role at [Company Name] and for taking the time to meet with us. After careful consideration, we regret to inform you that we have decided not to move forward with your candidacy for this position.",
                                        "",
                                        raw_feedback_body, # The core feedback from the AI
                                        "",
                                        "Although you were not selected for this particular opportunity, we encourage you to continue developing your skills in the mentioned areas. We appreciate your interest in [Company Name] and wish you the best of luck in your future career endeavors.",
                                        "",
                                        "Sincerely,",
                                        "The [Company Name] Hiring Team" # Or just "[Company Name]"
                                    ]
                                    feedback_email_body = "\n".join(full_email_parts)
                                    # --- END NEW CONSTRUCTION ---


                                    st.subheader("Personalized Feedback Email Preview:")
                                    st.text_area("Feedback Email Content", value=feedback_email_body, height=300)

                                    st.success("Feedback email generated successfully!")

                                    # --- Automatic Email Sending ---
                                    st.subheader("Automatic Email Sending Status:")
                                    recipient_email = extract_email_from_text(resume_content)
                                    
                                    if recipient_email:
                                        # Use a consistent subject line
                                        email_subject = f"Update on your Application for the Data Scientist Role at [Company Name]"
                                        st.info(f"Attempting to send email to {recipient_email} using SendGrid...")
                                        send_rejection_email(recipient_email, email_subject, feedback_email_body)
                                    else:
                                        st.warning("No email address found in the resume. Cannot send email automatically.")
                                    # --- END Automatic Email Sending ---

                                elif "error" in feedback_response_json:
                                    st.error(f"Failed to generate personalized feedback email: {feedback_response_json['error']}")
                                else:
                                    st.error("Failed to get a valid response from watsonx.ai for feedback generation.")
                            elif not feedback_prompt_template_content:
                                st.error("Feedback prompt template not loaded. Cannot generate feedback email.")
                            else: # This else covers cases where feedback_prompt_template_content is loaded but no rejection_reasons are found
                                st.warning("No rejection reasons found to generate feedback email (this should not happen for rejected resumes if parsing is correct).")

                    elif "error" in ats_response_json:
                        st.error(f"Failed to get ATS evaluation from watsonx.ai: {ats_response_json['error']}")
                    else:
                        st.error("Failed to get a valid response from watsonx.ai for ATS evaluation.")
            else:
                st.error(f"Could not extract content from {resume_file.name}. Please ensure it's a valid TXT or PDF.")
        st.markdown("---") # Separator for each resume result

st.markdown("---")
st.write("Powered by watsonx.ai")