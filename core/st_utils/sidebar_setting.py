import importlib.util
import time
import streamlit as st
import requests
from translations.translations import translate as t
from core.utils import *
from core.utils.ask_gpt import is_local_endpoint, normalize_base_url


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


def burned_subtitles_settings():
    """Which subtitles are burned into the subtitle video and into the dubbed video."""
    from core.utils.subtitle_style import BURN_MODES, BURN_MODE_KEYS, get_burn_mode
    labels = {
        "bilingual": t("Bilingual"),
        "translation": t("Translation only"),
        "source": t("Source only"),
    }
    columns = st.columns(2)
    for column, video, label in zip(columns, ("subtitle", "dubbed"), (t("Subtitle video"), t("Dubbed video"))):
        cfg_key = BURN_MODE_KEYS[video][0]
        mode = get_burn_mode(video)
        with column:
            selected = st.selectbox(
                label,
                options=list(BURN_MODES),
                index=BURN_MODES.index(mode) if mode in BURN_MODES else 0,
                format_func=lambda x: labels.get(x, x),
                help=t("Which subtitles are burned into this video"),
            )
        if selected != mode:
            update_key(cfg_key, selected, add_missing=True)
            st.rerun()


def subtitle_style_settings():
    """Font, size and color of the burned-in subtitles; the other values of subtitle.style are edited in config.yaml."""
    from core.utils.subtitle_style import default_font, get_subtitle_style, save_subtitle_style, to_ass_color, to_hex_color
    with st.popover(t("Subtitle Style"), use_container_width=True):
        style = {}
        columns = st.columns(2)
        for column, kind, label in zip(columns, ("source", "translation"), (t("Source subtitles"), t("Translated subtitles"))):
            current = get_subtitle_style(kind)
            with column:
                st.markdown(f"**{label}**")
                font_name = st.text_input(
                    t("Font"), value=current["font_name"], placeholder=default_font(), key=f"subtitle_font_{kind}",
                    help=t("Name of a font installed on this computer, empty for the default font"))
                font_size = st.number_input(
                    t("Font size"), min_value=1, max_value=100, value=int(current["font_size"]), key=f"subtitle_size_{kind}")
                color = st.color_picker(t("Font color"), value=to_hex_color(current["font_color"]), key=f"subtitle_color_{kind}")
            # The transparency of the configured color is kept as long as the color is
            font_color = current["font_color"] if color == to_hex_color(current["font_color"]) else to_ass_color(color)
            style[kind] = {**current, "font_name": font_name.strip(), "font_size": font_size, "font_color": font_color}
        if style != {kind: get_subtitle_style(kind) for kind in style}:
            save_subtitle_style(style)
            st.rerun()


# The languages shown first keep the order of the old dropdown
LANGUAGE_LABELS = {
    "en": "🇺🇸 English", "zh": "🇨🇳 简体中文", "es": "🇪🇸 Español", "ru": "🇷🇺 Русский",
    "fr": "🇫🇷 Français", "de": "🇩🇪 Deutsch", "it": "🇮🇹 Italiano", "ja": "🇯🇵 日本語",
    "yue": "🇭🇰 粵語", "ko": "🇰🇷 한국어", "pt": "🇵🇹 Português", "ar": "🇸🇦 العربية",
    "id": "🇮🇩 Bahasa Indonesia", "th": "🇹🇭 ไทย", "vi": "🇻🇳 Tiếng Việt", "tr": "🇹🇷 Türkçe",
    "hi": "🇮🇳 हिन्दी", "ms": "🇲🇾 Bahasa Melayu", "nl": "🇳🇱 Nederlands", "sv": "🇸🇪 Svenska",
    "da": "🇩🇰 Dansk", "fi": "🇫🇮 Suomi", "pl": "🇵🇱 Polski", "cs": "🇨🇿 Čeština",
    "fil": "🇵🇭 Filipino", "fa": "🇮🇷 فارسی", "el": "🇬🇷 Ελληνικά", "ro": "🇷🇴 Română",
    "hu": "🇭🇺 Magyar", "mk": "🇲🇰 Македонски",
}


def recognition_languages(configured=None):
    """Label -> code for the recognition language dropdown: every Qwen3-ASR language."""
    from core.asr_backend.qwen_asr_local import ISO_TO_QWEN

    langs = {t("Auto"): "auto"}
    for code in [*LANGUAGE_LABELS, *ISO_TO_QWEN]:
        if code in ISO_TO_QWEN and code not in langs.values():
            langs[LANGUAGE_LABELS.get(code, ISO_TO_QWEN[code])] = code
    # A code set by hand in config.yaml (WhisperX and ElevenLabs know more languages) stays selectable
    if configured not in langs.values():
        langs[str(configured)] = configured
    return langs


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not base_url or (not api_key and not is_local_endpoint(base_url)):
        return []
    url = normalize_base_url(base_url) + "/models"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted([m["id"] for m in data if "id" in m])
    except Exception:
        return []


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def page_setting():
    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    with st.expander(t("LLM Configuration"), expanded=True):
        from core.utils.llm_rotator import rotator

        # Preset providers quick selector
        st.markdown(f"**{t('Quick Presets')}**")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("⚡ Gemini", key="preset_gemini", use_container_width=True, help="Google Gemini 1.5 Flash (Free)"):
                update_key("api.base_url", "https://generativelanguage.googleapis.com/v1beta/openai")
                update_key("api.model", "gemini-1.5-flash")
                update_key("api.llm_support_json", True)
                rotator.set_auth_mode("gemini_accounts")
                st.rerun()
        with p_col2:
            if st.button("🌟 DeepSeek", key="preset_deepseek", use_container_width=True, help="DeepSeek-Chat"):
                update_key("api.base_url", "https://api.deepseek.com/v1")
                update_key("api.model", "deepseek-chat")
                update_key("api.llm_support_json", True)
                rotator.set_auth_mode("key_pool")
                st.rerun()
        with p_col3:
            if st.button("🤖 OpenAI", key="preset_openai", use_container_width=True, help="GPT-4o mini"):
                update_key("api.base_url", "https://api.openai.com/v1")
                update_key("api.model", "gpt-4o-mini")
                update_key("api.llm_support_json", True)
                rotator.set_auth_mode("key_pool")
                st.rerun()

        # Connection Mode selection
        auth_mode_options = {
            "key_pool": t("🔑 Danh sách API Key (Xoay vòng)"),
            "gemini_accounts": t("👤 Tài khoản Gemini (Nhiều tài khoản)"),
        }
        current_auth_mode = rotator.get_auth_mode()
        selected_auth_mode = st.radio(
            t("Phương thức kết nối AI"),
            options=list(auth_mode_options.keys()),
            format_func=lambda x: auth_mode_options.get(x, x),
            index=list(auth_mode_options.keys()).index(current_auth_mode) if current_auth_mode in auth_mode_options else 0,
            horizontal=True,
        )
        if selected_auth_mode != current_auth_mode:
            rotator.set_auth_mode(selected_auth_mode)
            st.rerun()

        if selected_auth_mode == "gemini_accounts":
            # Gemini Accounts Manager
            accounts = rotator.get_gemini_accounts()
            st.markdown(f"**{t('Danh sách tài khoản Google / Gemini')}:** ({len(accounts)} {t('tài khoản')})")
            
            for a_idx, acc in enumerate(accounts):
                a_col1, a_col2 = st.columns([4, 1])
                with a_col1:
                    masked = rotator.mask_key(acc.get("key", ""))
                    label = acc.get("label") or acc.get("email") or f"Tài khoản {a_idx + 1}"
                    st.markdown(f"👤 **{label}** (`{masked}`)")
                with a_col2:
                    if st.button("🗑️", key=f"del_gemini_acc_{a_idx}"):
                        accounts.pop(a_idx)
                        rotator.set_gemini_accounts(accounts)
                        st.rerun()

            st.markdown(
                """
                <div style="background: rgba(66, 133, 244, 0.08); border: 1px solid rgba(66, 133, 244, 0.3); border-radius: 6px; padding: 10px 12px; margin: 8px 0; font-size: 0.85rem; line-height: 1.5;">
                    💡 <b>Cách lấy Khóa API Gemini chính thức (Miễn phí 100%):</b><br>
                    1. Mở trang Google AI Studio: <a href="https://aistudio.google.com/apikey" target="_blank" style="color: #4285F4; font-weight: bold; text-decoration: underline;">https://aistudio.google.com/apikey</a><br>
                    2. Bấm <b>"Create API key"</b> (Tạo khóa) &rarr; Copy chuỗi mã <code>AIzaSy...</code> dán vào ô bên dưới.<br>
                    <i>(Mỗi tài khoản Google được cấp miễn phí 1.500 lượt/ngày. Dùng API chính thức sẽ an toàn tuyệt đối, không sợ bị khóa tài khoản như dùng cookie web!)</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(t("➕ Thêm tài khoản Gemini mới"), expanded=(len(accounts) == 0)):
                new_acc_label = st.text_input(t("Tên hoặc Email tài khoản:"), placeholder="user1@gmail.com", key="input_gemini_email")
                new_acc_key = st.text_input(t("Khóa API Gemini của tài khoản này:"), type="password", placeholder="AIzaSy...", key="input_gemini_key", help=t("Lấy miễn phí tại https://aistudio.google.com/apikey"))
                if st.button(t("Lưu tài khoản vào danh sách xoay vòng"), use_container_width=True, key="save_gemini_acc"):
                    if new_acc_key.strip():
                        accounts.append({
                            "email": new_acc_label.strip() or f"Tài khoản {len(accounts) + 1}",
                            "label": new_acc_label.strip() or f"Tài khoản {len(accounts) + 1}",
                            "key": new_acc_key.strip()
                        })
                        rotator.set_gemini_accounts(accounts)
                        st.toast(t("Đã thêm tài khoản thành công!"), icon="✅")
                        st.rerun()
                    else:
                        st.warning(t("Vui lòng nhập Khóa API của tài khoản!"))

        else:
            # Multi-Key Pool Mode
            pool_keys = rotator.get_pool_keys()
            keys_text = "\n".join(pool_keys)
            entered_keys_text = st.text_area(
                t("Danh sách API Key (mỗi dòng 1 Key để tự động xoay):"),
                value=keys_text,
                placeholder=t("Dán một hoặc nhiều API Key tại đây, mỗi dòng một Key..."),
                help=t("Hệ thống sẽ dùng Key đầu tiên. Khi Key đó hết lượt/gặp lỗi (429 RateLimit), hệ thống sẽ tự động chuyển sang Key tiếp theo."),
                height=110,
            )
            parsed_keys = [k.strip() for k in entered_keys_text.replace(",", "\n").split("\n") if k.strip()]
            if parsed_keys != pool_keys:
                rotator.set_pool_keys(parsed_keys)
                st.rerun()

            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #888; margin-top: -6px; margin-bottom: 8px;">
                    🔗 Lấy API Key Gemini miễn phí tại: <a href="https://aistudio.google.com/apikey" target="_blank" style="color: #4285F4;">aistudio.google.com/apikey</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pool_keys:
                st.caption(f"💡 {t('Đang nạp')} **{len(pool_keys)}** {t('API Key trong danh sách xoay vòng.')}")

        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", use_container_width=True
            ):
                with st.spinner(t("Fetching models...")):
                    active_key, _, _ = rotator.get_active_credential()
                    models = _fetch_model_list(
                        load_key("api.base_url"), active_key
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    placeholder=t("Search or enter model name"),
                )

        auto_rotate = st.toggle(
            t("Tự động xoay vòng khi hết Token / Lỗi (Auto-rotate)"),
            value=rotator.is_auto_rotate_enabled(),
            help=t("Tự động chuyển sang tài khoản hoặc API Key tiếp theo khi gặp mã lỗi 429 (hết token), 403 hoặc lỗi máy chủ."),
        )
        if auto_rotate != rotator.is_auto_rotate_enabled():
            rotator.set_auto_rotate(auto_rotate)
            st.rerun()

        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

        if st.button("📡 " + t("Kiểm tra tất cả Key / Tài khoản"), key="check_all_creds_btn", use_container_width=True):
            creds = rotator.get_all_credentials()
            if not creds:
                st.warning(t("Chưa có API Key hoặc Tài khoản nào được cấu hình!"))
            else:
                base_u = load_key("api.base_url")
                m_name = load_key("api.model")
                with st.spinner(t("Đang kiểm tra từng tài khoản / API Key...")):
                    for c in creds:
                        ok, msg = rotator.test_single_key(c["key"], base_u, m_name)
                        if ok:
                            st.success(f"✅ **{c['label']}** ({c['masked']}): {msg}")
                        else:
                            st.error(f"❌ **{c['label']}** ({c['masked']}): {msg}")
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = recognition_languages(load_key("whisper.language"))
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtimes = ["local", "elevenlabs", "mai"]
        configured_runtime = load_key("whisper.runtime")
        if configured_runtime not in runtimes:
            st.warning(t("The 302.ai WhisperX cloud service has been retired. Select Local or ElevenLabs to continue."))
        runtime = st.selectbox(
            t("ASR Runtime"),
            options=runtimes,
            index=runtimes.index(configured_runtime) if configured_runtime in runtimes else None,
            format_func=lambda x: {
                "local": t("Local"),
                "elevenlabs": t("ElevenLabs"),
                "mai": t("MAI-Transcribe-2"),
            }.get(x, x),
            help=t(
                "Local Qwen3-ASR runs best on an NVIDIA GPU or Apple Silicon (CPU works but is slow); cloud recognition requires a provider API key."
            ),
        )
        if runtime is not None and runtime != configured_runtime:
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "local":
            backends = ["qwen", "whisperx"]
            configured_backend = load_key_or("whisper.backend", "qwen")
            backend = st.selectbox(
                t("Local ASR Backend"),
                options=backends,
                index=backends.index(configured_backend) if configured_backend in backends else 0,
                format_func=lambda x: {
                    "qwen": t("Qwen3-ASR + ForcedAligner (default)"),
                    "whisperx": t("WhisperX (manual install)"),
                }.get(x, x),
            )
            if backend != configured_backend:
                update_key("whisper.backend", backend, add_missing=True)
                st.rerun()
            if backend == "whisperx" and importlib.util.find_spec("whisperx") is None:
                st.warning(t("WhisperX is not installed. Follow the manual page (docs/pages/docs/whisperx-manual.en-US.md) and install the extra packages yourself, or set whisper.backend to qwen."))
        if runtime == "elevenlabs":
            config_input(t("ElevenLabs API"), "whisper.elevenlabs_api_key")
        elif runtime == "mai":
            from core.asr_backend.mai_asr import detect_region

            configured_provider = str(load_key_or("whisper.mai_provider", "azure") or "azure").lower()
            providers = ["azure", "openrouter"]
            provider = st.selectbox(
                t("MAI provider"), options=providers,
                index=providers.index(configured_provider) if configured_provider in providers else 0,
                format_func=lambda value: "Azure Speech" if value == "azure" else "OpenRouter",
            )
            if provider != configured_provider:
                update_key("whisper.mai_provider", provider, add_missing=True)
                st.rerun()
            if provider == "openrouter":
                stored_key = str(load_key_or("whisper.mai_openrouter_api_key", "") or "")
                router_key = st.text_input(
                    t("OpenRouter API key"), value=stored_key, type="password",
                )
                if router_key != stored_key:
                    update_key("whisper.mai_openrouter_api_key", router_key, add_missing=True)
                st.caption(t("Audio is sent to OpenRouter for MAI-Transcribe-2 and may incur charges."))
            else:
                stored_key = str(load_key_or("whisper.mai_api_key", "") or "")
                speech_key = st.text_input(
                    t("Azure Speech key"), value=stored_key, type="password",
                    help=t("Use a key from an Azure Speech resource in a region where MAI-Transcribe-2 is available."),
                )
                if speech_key != stored_key:
                    update_key("whisper.mai_api_key", speech_key, add_missing=True)
                stored_region = str(load_key_or("whisper.mai_region", "") or "")
                region = st.text_input(
                    t("Azure Speech region or resource endpoint"), value=stored_region,
                    help=t("Enter a region such as eastus, or the HTTPS endpoint of your Azure Speech resource. Leave blank to detect the region from the key."),
                )
                if region != stored_region:
                    update_key("whisper.mai_region", region.strip(), add_missing=True)
                if st.button(t("Detect Azure Speech region"), disabled=not speech_key.strip()):
                    with st.spinner(t("Detecting Azure Speech region...")):
                        detected = detect_region(speech_key.strip())
                    if detected:
                        update_key("whisper.mai_region", detected, add_missing=True)
                        st.rerun()
                    else:
                        st.warning(t("Region detection failed. Check the key or enter the resource region manually."))
                st.caption(t("MAI-Transcribe-2 is in public preview. Audio is sent to Azure Speech."))

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        configured_style = str(load_key_or("translation_style", "") or "")
        translation_style = st.text_input(
            t("Translation style"),
            value=configured_style,
            placeholder=t("e.g. colloquial, short sentences"),
            help=t("Optional. Describe the style of the translation in your own words, it is added to the translation prompt"),
        )
        if translation_style != configured_style:
            update_key("translation_style", translation_style, add_missing=True)
            st.rerun()

        demucs_available = importlib.util.find_spec("demucs") is not None
        if not demucs_available and load_key("demucs"):
            update_key("demucs", False)
        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            disabled=not demucs_available,
            help=t("Vocal separation is unavailable in this installation") if not demucs_available else t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        from core._1_ytdlp import is_audio_only_input
        audio_only = is_audio_only_input()
        if audio_only:
            st.toggle(
                t("Burn-in Subtitles"),
                value=False,
                disabled=True,
                help=t("Audio-only input produces subtitle files only; no video is generated."),
            )
        else:
            burn_subtitles = st.toggle(
                t("Burn-in Subtitles"),
                value=load_key("burn_subtitles"),
                help=t(
                    "Whether to burn subtitles into the video, will increase processing time"
                ),
            )
            if burn_subtitles != load_key("burn_subtitles"):
                update_key("burn_subtitles", burn_subtitles)
                st.rerun()
            if burn_subtitles:
                burned_subtitles_settings()
                subtitle_style_settings()

        pause_before_translate = st.toggle(
            t("Pause before translation"),
            value=load_key("pause_before_translate"),
            help=t("Pause after the terminology is extracted, so that you can edit `output/log/terminology.json` before the translation starts"),
        )
        if pause_before_translate != load_key("pause_before_translate"):
            update_key("pause_before_translate", pause_before_translate)
            st.rerun()

        pause_after_translate = st.toggle(
            t("Pause after translation"),
            value=bool(load_key_or("pause_after_translate", False)),
            help=t("Pause after the translation, so that you can edit the `Translation` column of `output/log/translation_results.xlsx` before the subtitles are generated"),
        )
        if pause_after_translate != bool(load_key_or("pause_after_translate", False)):
            update_key("pause_after_translate", pause_after_translate, add_missing=True)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_method_labels = {
            "openai_tts": t("OpenAI TTS"),
            "fish_audio": t("Fish Audio"),
            "sf_fish_tts": t("SiliconFlow Fish TTS"),
            "edge_tts": t("Edge TTS"),
            "gpt_sovits": t("GPT-SoVITS"),
            "custom_tts": t("Custom TTS"),
            "sf_cosyvoice2": t("SiliconFlow CosyVoice2"),
            "f5tts": t("F5-TTS"),
        }
        from core.st_utils.tts_settings import select_tts_method
        select_tts = select_tts_method(tts_method_labels)

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options.get(x, str(x)),
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input(t("Voice"), "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_audio":
            from core.st_utils.tts_settings import save_settings
            from core.tts_backend.fish_audio_tts import voice_id_of
            mode_options = {
                "clone": t("Voice of the video"),
                "preset": t("Fixed voice"),
            }
            current_mode = load_key_or("fish_audio.mode", "clone")
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options.get(x, str(x)),
                index=list(mode_options.keys()).index(current_mode) if current_mode in mode_options else 0,
            )
            if selected_mode != current_mode:
                save_settings({"fish_audio.mode": selected_mode})
                st.rerun()
            if selected_mode == "preset":
                voices = dict(load_key_or("fish_audio.voices", None) or {})
                voice_id = str(load_key_or("fish_audio.voice_id", "") or "")
                other = t("Another voice of fish.audio")
                names = list(voices) + [other]
                listed = next((name for name, listed_id in voices.items() if listed_id == voice_id), other)
                # The key keeps "Another voice" selected while its field is still empty
                selected_voice = st.selectbox(t("Voice"), names, index=names.index(listed), key="fish_audio_voice")
                if selected_voice == other:
                    entered = st.text_input(t("Voice ID or the address of its page"),
                                            value="" if voice_id in voices.values() else voice_id,
                                            help=t("Open a voice on fish.audio and copy the address of its page"))
                    selected_id = voice_id_of(entered)
                else:
                    selected_id = voices[selected_voice]
                if selected_id and selected_id != voice_id:
                    save_settings({"fish_audio.voice_id": selected_id})

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options.get(x, str(x)),
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")



def check_api():
    """Returns (is_valid, error): one request without retries, never answered from the cache."""
    try:
        resp = ask_gpt.__wrapped__(
            f"This is a test ({time.time():.0f}), response 'message':'success' in json format.",
            resp_type="json",
            log_title="None",
        )
        if resp.get("message") == "success":
            return True, ""
        return False, f"Unexpected response: {str(resp)[:300]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:500]}"


def show_api_check(is_valid, error):
    if is_valid:
        st.toast(t("API Key is valid"), icon="✅")
    else:
        st.toast(t("API check failed"), icon="❌")
        st.error(f"{t('API check failed')}: {error}")


if __name__ == "__main__":
    check_api()
