#!/usr/bin/env python3
"""
Script to send JSON request files to sglang server
"""
import json
import argparse
import requests
from pathlib import Path


def send_request_to_sglang(
    request_file: str,
    sglang_url: str = "http://localhost:8000",
    output_file: str = None,
    verbose: bool = False
):
    """
    Send a JSON request file to sglang server

    Args:
        request_file: Path to the JSON request file
        sglang_url: Base URL of the sglang server (default: http://localhost:8000)
        output_file: Optional path to save the response
        verbose: Print detailed information
    """
    # Read the request JSON
    request_path = Path(request_file)
    if not request_path.exists():
        raise FileNotFoundError(f"Request file not found: {request_file}")

    with open(request_path, 'r') as f:
        request_data = json.load(f)

    if verbose:
        print(f"Loading request from: {request_file}")
        print(f"Model: {request_data.get('model', 'N/A')}")
        print(f"Messages: {len(request_data.get('messages', []))}")

        # Count images in the request
        image_count = 0
        for msg in request_data.get('messages', []):
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'image_url' in item:
                        image_count += 1

        if image_count > 0:
            print(f"Images detected: {image_count}")

        print(f"Sending to: {sglang_url}/v1/chat/completions")

    # Prepare the endpoint
    endpoint = f"{sglang_url.rstrip('/')}/v1/chat/completions"

    # Send the request
    try:
        print("Sending request to sglang...")
        response = requests.post(
            endpoint,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=600  # 10 minute timeout
        )
        response.raise_for_status()

        response_data = response.json()

        if verbose:
            print("\n" + "="*50)
            print("Response received successfully!")
            print("="*50)

        # Print the response
        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0].get('message', {}).get('content', '')
            print("\nModel Response:")
            print("-" * 50)
            print(content)
            print("-" * 50)
        else:
            print("\nFull Response:")
            print(json.dumps(response_data, indent=2))

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(response_data, f, indent=2)

            print(f"\nResponse saved to: {output_file}")

        return response_data

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to sglang server at {sglang_url}")
        print("Make sure the sglang server is running!")
        raise
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 10 minutes")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        raise
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise


def main():
    # Default request file location
    DEFAULT_REQUEST_FILE = '/vast/projects/liuv/pennnetworks/jiaheng/BrowserUseScript/agent_logs/hello_world/vllm_requests/request_002_20251126_155624/request.json'

    parser = argparse.ArgumentParser(
        description='Send JSON request files to sglang server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default request file
  python send_sglang_request.py

  # Specify a custom request file
  python send_sglang_request.py /path/to/request.json

  # Specify sglang server URL
  python send_sglang_request.py --url http://localhost:8080

  # Save response to file
  python send_sglang_request.py --output response.json

  # Verbose mode
  python send_sglang_request.py -v
        """
    )

    parser.add_argument(
        'request_file',
        type=str,
        nargs='?',
        default=DEFAULT_REQUEST_FILE,
        help=f'Path to the JSON request file (default: {DEFAULT_REQUEST_FILE})'
    )

    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='sglang server URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save the response JSON'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    try:
        send_request_to_sglang(
            request_file=args.request_file,
            sglang_url=args.url,
            output_file=args.output,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nScript failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
