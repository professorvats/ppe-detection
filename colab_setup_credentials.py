#!/usr/bin/env python3
"""
=============================================================================
CREDENTIALS SETUP FOR COLAB - RUN THIS FIRST
=============================================================================
This cell sets up AWS and OpenAI credentials for the PPE research pipeline.
Run this cell ONCE at the start of your Colab session.

SECURITY NOTE:
- Never share this file with credentials filled in
- Use Colab Secrets (recommended) or paste credentials when prompted
=============================================================================
"""

import os

# ============================================================================
# OPTION 1: Set credentials directly (for quick testing only)
# ============================================================================
# Uncomment and fill in your credentials:

os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_HERE'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_KEY_HERE'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY_HERE'

# ============================================================================
# OPTION 2: Use Colab Secrets (RECOMMENDED - More Secure)
# ============================================================================
# In Colab: Click the key icon on the left sidebar > Add secrets
# Then uncomment below:

# from google.colab import userdata
# os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
# os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
# os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# ============================================================================
# OPTION 3: Prompt for credentials (Interactive)
# ============================================================================
# Uncomment below to enter credentials interactively:

# import getpass
# os.environ['AWS_ACCESS_KEY_ID'] = getpass.getpass('AWS Access Key ID: ')
# os.environ['AWS_SECRET_ACCESS_KEY'] = getpass.getpass('AWS Secret Key: ')
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key: ')

# ============================================================================
# Verify credentials are set
# ============================================================================
def verify_credentials():
    """Check if credentials are configured"""
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
    openai_key = os.environ.get('OPENAI_API_KEY', '')

    print("\n" + "="*50)
    print("  CREDENTIALS STATUS")
    print("="*50)
    print(f"  AWS_ACCESS_KEY_ID:     {'✓ Set' if aws_key and aws_key != 'YOUR_AWS_ACCESS_KEY_HERE' else '✗ Not set'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'✓ Set' if aws_secret and aws_secret != 'YOUR_AWS_SECRET_KEY_HERE' else '✗ Not set'}")
    print(f"  OPENAI_API_KEY:        {'✓ Set' if openai_key and openai_key != 'YOUR_OPENAI_KEY_HERE' else '✗ Not set'}")
    print("="*50 + "\n")

    return all([
        aws_key and aws_key != 'YOUR_AWS_ACCESS_KEY_HERE',
        aws_secret and aws_secret != 'YOUR_AWS_SECRET_KEY_HERE',
    ])

if __name__ == '__main__':
    verify_credentials()
