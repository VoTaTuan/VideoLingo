import os
import streamlit as st
import io, zipfile
from core.st_utils.download_video_section import download_video_section
from core.st_utils.sidebar_setting import page_setting
from translations.translations import translate as t

def download_subtitle_zip_button(text: str):
    zip_buffer = io.BytesIO()
    output_dir = "output"
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".srt"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "rb") as file:
                    zip_file.writestr(file_name, file.read())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label=text,
        data=zip_buffer,
        file_name="subtitles.zip",
        mime="application/zip"
    )

# st.markdown
_GIVE_STAR_BUTTON_TEMPLATE = """
<style>
    .github-button {
        display: block;
        width: 100%;
        padding: 0.5em 1em;
        color: #144070;
        background-color: #d0e0f2;
        border-radius: 6px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        transition: background-color 0.3s ease, color 0.3s ease;
        box-sizing: border-box;
    }
    .github-button:hover {
        background-color: #ffffff;
        color: #144070;
    }
</style>
<a href="https://github.com/Huanshere/VideoLingo" target="_blank" style="text-decoration: none;">
    <div class="github-button">
        __STAR_LABEL__
    </div>
</a>
"""


def give_star_button():
    return _GIVE_STAR_BUTTON_TEMPLATE.replace("__STAR_LABEL__", t("Star on GitHub 🌟"))

button_style = """
<style>
div.stButton > button:first-child,
div.stDownloadButton > button:first-child {
    display: block;
    width: 100%;
    padding: 0.55em 1.2em;
    color: #1a56db;
    background-color: #f8faff;
    text-decoration: none;
    font-weight: 600;
    text-align: center;
    transition: all 0.15s ease-in-out;
    box-sizing: border-box;
    border: 2px solid #b8d5fb;
    border-radius: 8px;
    font-size: 1.1em;
    cursor: pointer !important;
    user-select: none;
}
div.stButton > button:first-child:hover,
div.stDownloadButton > button:first-child:hover {
    background-color: #eaf2ff !important;
    color: #1043a6 !important;
    border-color: #79aef7 !important;
    box-shadow: 0 2px 8px rgba(26, 86, 219, 0.18) !important;
}
div.stButton > button:first-child:active,
div.stDownloadButton > button:first-child:active {
    transform: scale(0.98) !important;
    background-color: #d8e8fe !important;
    border-color: #1a56db !important;
    box-shadow: 0 1px 2px rgba(26, 86, 219, 0.2) !important;
}
div.stButton > button:first-child:focus,
div.stDownloadButton > button:first-child:focus {
    outline: none !important;
    border-color: #1a56db !important;
}
/* Dark theme support */
@media (prefers-color-scheme: dark) {
    div.stButton > button:first-child,
    div.stDownloadButton > button:first-child {
        color: #93c5fd !important;
        background-color: #1e293b !important;
        border-color: #3b82f6 !important;
    }
    div.stButton > button:first-child:hover,
    div.stDownloadButton > button:first-child:hover {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #60a5fa !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }
    div.stButton > button:first-child:active,
    div.stDownloadButton > button:first-child:active {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
}
</style>
"""
