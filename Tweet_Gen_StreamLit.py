import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import base64
from datetime import datetime


party_dict = {
    'AfD':'/Users/ianfischer/Documents/Uni/streamlit/models/afd_gpt2',
    'Die Grünen':'/Users/ianfischer/Documents/Uni/streamlit/models/gruene_gpt2',
    'CDU/CSU':'/Users/ianfischer/Documents/Uni/streamlit/models/cdu_csu_gpt2',
    'die Linke':'/Users/ianfischer/Documents/Uni/streamlit/models/linke_gpt2', 
}

picture_paths = {
    'blue_check': "/streamlit/app_pictures/Twitter_Verified_Badge.svg.png",
    'AfD' : "/streamlit/app_pictures/afd_logo.png",
    'CDU/CSU' : "/streamlit/app_pictures/cdu_logo.png",
    'Die Grünen' : "/streamlit/app_pictures/gruene_logo.png",
    'die Linke' : "/streamlit/app_pictures/linke_logo.png"
}

party_info = {
    "AfD": ("AfD", "@AfD"),
    "CDU/CSU": ("CDU", "@CDU"),
    "Die Grünen": ("Die Grünen", "@Die_Gruenen"),
    "die Linke": ("Die Linke", "@dieLinke")
}

def generate_tweet(party, prompt):
    device = torch.device("cpu")
    model_path = party_dict[party]
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.to(device)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        top_k=20,
        max_length=280,
        top_p=0.95,
        num_return_sequences=1,
        temperature=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tweet = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return generated_tweet
    
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')

def main():
    st.title("Tweet GPT")
    st.sidebar.header("**Analyzing Rhetorical Styles of German Political Parties**")
    st.sidebar.write("This project analyzes rhetorical differences among major German parties by generating tweets tailored to their communication styles. Our motivation was to understand the nuances in language and messaging strategies, particularly the effective use of social media by right-wing parties to disseminate messages. We aimed to raise awareness of this issue and contribute to a better understanding of varying rhetorical approaches.")
    st.sidebar.markdown("#### Motivation")
    st.sidebar.markdown("- Understand language and messaging nuances")
    st.sidebar.markdown("- Explore rhetorical differences and communication styles")
    st.sidebar.markdown("- Provide insights into crafting messages for supporters")
    st.sidebar.markdown("- Raise awareness of right-wing parties' social media use")
    st.sidebar.write("Reference: https://www.zdf.de/nachrichten/politik/deutschland/afd-tiktok-erfolg-strategie-jugendliche-100.html")

    tweet = ""
    prompt = ""

    col1, col2, col3 = st.columns([14, 1, 14])

    with col1:
        st.write("### Generate Tweets!")
        party = st.selectbox("Party", ["AfD", "CDU/CSU", "Die Grünen", "die Linke"], key='party')
        if st.button("Generate Tweet", key='generate', help="Click here to generate the tweet."):
            prompt = st.session_state.get("prompt", "")
            tweet = generate_tweet(party, prompt)

    with col2: 
        st.write("")
    
    with col3:
        st.write("### Topic")
        prompt = st.text_area("Tweet Keyword", key="prompt")

    with st.container():
        current_date = datetime.now().strftime("%B %d, %Y")
        blue_check_base64 = get_base64_image(picture_paths['blue_check'])
        party_logo_base64 = get_base64_image(picture_paths[party])
        party, username = party_info[party]

        if tweet:
            tweet_display = f'''
            <div style="background-color: white; padding: 10px; font-family: Helvetica Neue, sans-serif; border: 1px solid #ccc; color: black; border-radius: 10px;">
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{party_logo_base64}" alt="{party} Logo" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
                    <div style="display: flex; align-items: center;">
                        <div style="display: flex; align-items: center; margin-right: 5px;">
                             <p style="font-weight: bold; color: black;">{party}<img src="data:image/png;base64,{blue_check_base64}" alt="checkmark" style="width: 20px; height: 20px; vertical-align: middle; margin-left: 2px; margin-right: 2px;"><span style="font-weight: normal; color: gray;"> {username}</span> <span style="font-weight: normal; color: gray;">{current_date}</span></p>
                </div>
                    </div>
                </div>
                <p style="color: black;">{tweet}</p>
            </div>
            '''
            st.markdown(tweet_display, unsafe_allow_html=True)

        st.write("## Our Insights")
        st.write("We developed a tweet generator that analyzes the dataset for each party. To view the insights, select the button to see the most common topics each party wrote about.")
        

if __name__ == "__main__":
    main()
