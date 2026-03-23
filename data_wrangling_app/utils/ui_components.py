import streamlit as st

def render_glass_card(text: str, icon_svg: str, rotation: int = 0):
    """Renders a modern glassmorphism card from LayoutUI/cat_one.py."""
    st.markdown(f"""
    <div class="glass-container">
      <div data-text="{text}" style="--r:{rotation};" class="glass">
        {icon_svg}
      </div>
    </div>
    <style>
    .glass-container {{ position: relative; display: flex; justify-content: center; align-items: center; margin: 20px 0; }}
    .glass {{ 
        position: relative; width: 180px; height: 180px; 
        background: linear-gradient(#fff2, transparent); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        box-shadow: 0 25px 25px rgba(0, 0, 0, 0.25); 
        display: flex; justify-content: center; align-items: center; 
        transition: 0.5s; border-radius: 10px; backdrop-filter: blur(10px); 
        transform: rotate(calc(var(--r) * 1deg)); 
    }}
    .glass:hover {{ transform: rotate(0deg); margin: 0 10px; }}
    .glass::before {{ 
        content: attr(data-text); position: absolute; bottom: 0; 
        width: 100%; height: 40px; background: rgba(255, 255, 255, 0.05); 
        display: flex; justify-content: center; align-items: center; color: #fff; 
        font-family: sans-serif; font-weight: bold;
    }}
    .glass svg {{ font-size: 2.5em; fill: #fff; }}
    </style>
    """, unsafe_allow_html=True)

def render_tooltip(title: str, content: str, main_text: str):
    """Renders a Material Design 3 tooltip from LayoutUI/cat_three.py."""
    st.markdown(f"""
    <div class="tooltip-container">
      <span class="tooltip">
        <p class="tooltip-title">{title}</p>
        <p class="tooltip-content">{content}</p>
      </span>
      <span class="main-text">{main_text}</span>
    </div>
    <style>
    .tooltip-container {{ 
        position: relative; background: #e8def8; color: #1d192b; cursor: pointer; 
        transition: all 0.2s; font-size: 16px; padding: 0.5em 1.5em; 
        border-radius: 50px; display: inline-block; margin: 5px;
    }}
    .tooltip-container .main-text {{ font-weight: bold; }}
    .tooltip {{ 
        transform-origin: center bottom; scale: 0; position: absolute; 
        bottom: 130%; left: 50%; transform: translate(-50%, 10px); 
        transition: all 0.2s; background: #f3edf7; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); padding: 15px; 
        border-radius: 12px; color: #49454f; min-width: 250px; z-index: 1000;
    }}
    .tooltip-container:hover .tooltip {{ scale: 1; transform: translate(-50%, 0); }}
    .tooltip-title {{ font-weight: bold; margin: 0 0 5px 0; }}
    .tooltip-content {{ font-size: 0.9em; margin: 0; }}
    </style>
    """, unsafe_allow_html=True)

def render_title_card(title: str, subtitle: str, tag: str = "Premium"):
    """Renders a clean title card from LayoutUI/cat_two.py."""
    st.markdown(f"""
    <article class="ui-card">
      <div class="ui-card-img">
        <div class="ui-card-accent"></div>
      </div>
      <div class="ui-project-info">
        <div class="ui-flex">
          <div class="ui-project-title">{title}</div>
          <span class="ui-tag">{tag}</span>
        </div>
        <span class="ui-lighter">{subtitle}</span>
      </div>
    </article>
    <style>
    .ui-card {{ 
        background-color: white; color: black; width: 100%; max-width: 350px; 
        border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
        margin: 20px auto; overflow: visible; font-family: sans-serif;
    }}
    .ui-card-img {{ position: relative; top: -15px; height: 80px; display: flex; justify-content: center; }}
    .ui-card-accent {{ height: 120px; width: 90%; background-color: #b2b2fd; border-radius: 8px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); }}
    .ui-project-info {{ padding: 60px 25px 25px 25px; display: flex; flex-direction: column; gap: 10px; }}
    .ui-project-title {{ font-weight: 600; font-size: 1.3em; color: #1a1a1a; }}
    .ui-tag {{ font-weight: 300; color: #666; font-size: 0.8em; text-transform: uppercase; }}
    .ui-lighter {{ font-size: 0.9em; color: #444; line-height: 1.4; }}
    .ui-flex {{ display: flex; justify-content: space-between; align-items: center; }}
    </style>
    """, unsafe_allow_html=True)
