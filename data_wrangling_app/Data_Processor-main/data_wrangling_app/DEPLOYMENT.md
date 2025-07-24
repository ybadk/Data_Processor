# 🚀 Deployment Guide

This guide will help you deploy the Data Wrangling & Analytics Platform to various platforms.

## 📋 Pre-Deployment Checklist

1. **Test the application locally**
   ```bash
   python test_app.py
   streamlit run app.py
   ```

2. **Verify all dependencies are listed in requirements.txt**

3. **Set up environment variables (if needed)**
   - EMAIL_USER
   - EMAIL_PASSWORD
   - SECRET_KEY

## 🌐 Streamlit Cloud Deployment

### Step 1: Prepare Repository
1. Create a GitHub repository
2. Push all files to the repository
3. Ensure the repository is public or you have Streamlit Cloud access

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set the main file path: `app.py`
6. Click "Deploy"

### Step 3: Configure Environment Variables (Optional)
1. In Streamlit Cloud dashboard, go to your app settings
2. Add environment variables in the "Secrets" section:
   ```toml
   EMAIL_USER = "your-email@gmail.com"
   EMAIL_PASSWORD = "your-app-password"
   SECRET_KEY = "your-secret-key"
   ```

## 🐳 Docker Deployment

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run
```bash
# Build the image
docker build -t data-wrangling-app .

# Run the container
docker run -p 8501:8501 data-wrangling-app
```

## ☁️ Heroku Deployment

### Step 1: Install Heroku CLI
Download and install from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

### Step 2: Prepare Files
Ensure you have:
- `Procfile`
- `setup.sh`
- `requirements.txt`

### Step 3: Deploy
```bash
# Login to Heroku
heroku login

# Create a new app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Step 4: Configure Environment Variables
```bash
heroku config:set EMAIL_USER=your-email@gmail.com
heroku config:set EMAIL_PASSWORD=your-app-password
heroku config:set SECRET_KEY=your-secret-key
```

## 🖥️ Local Production Deployment

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run in Production Mode
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Step 3: Use Process Manager (Optional)
```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
echo 'module.exports = {
  apps: [{
    name: "data-wrangling-app",
    script: "streamlit",
    args: "run app.py --server.port 8501 --server.address 0.0.0.0",
    interpreter: "python3"
  }]
}' > ecosystem.config.js

# Start with PM2
pm2 start ecosystem.config.js
```

## 🔧 Environment Configuration

### Production Environment Variables
```bash
# Email configuration
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password

# Security
SECRET_KEY=your-secret-key-here

# Database (if using external DB)
DATABASE_URL=sqlite:///data_wrangling.db

# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#000000"
secondaryBackgroundColor = "#1a1a1a"
textColor = "#FFFFFF"
```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit sensitive data to version control
2. **HTTPS**: Use HTTPS in production (handled by most cloud platforms)
3. **Database**: Secure your database with proper authentication
4. **Email**: Use app-specific passwords for email services
5. **File Uploads**: Implement file size and type restrictions

## 📊 Monitoring and Maintenance

### Health Checks
- Monitor application uptime
- Check database connectivity
- Verify email service functionality

### Performance Monitoring
- Track response times
- Monitor memory usage
- Check for errors in logs

### Regular Updates
- Update dependencies regularly
- Monitor for security vulnerabilities
- Backup database regularly

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **Database Errors**
   - Check file permissions
   - Verify SQLite installation

4. **Email Not Working**
   - Verify SMTP settings
   - Check app-specific passwords
   - Test email connectivity

### Getting Help
- Check application logs
- Review error messages
- Contact support: kgothatsothooe@gmail.com

## 📈 Scaling Considerations

For high-traffic deployments:
1. Use external database (PostgreSQL, MySQL)
2. Implement caching (Redis)
3. Use load balancers
4. Consider container orchestration (Kubernetes)

---

**Need help with deployment? Contact Profit Projects Online Virtual Assistance**
📧 kgothatsothooe@gmail.com
