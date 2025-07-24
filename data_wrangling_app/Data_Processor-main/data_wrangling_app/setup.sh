mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"kgothatsothooe@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
[theme]\n\
primaryColor = \"#FF6B6B\"\n\
backgroundColor = \"#000000\"\n\
secondaryBackgroundColor = \"#1a1a1a\"\n\
textColor = \"#FFFFFF\"\n\
" > ~/.streamlit/config.toml
