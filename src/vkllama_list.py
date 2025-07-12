import json
import requests
import datetime

VKLLAMA_MODELS_URL = 'http://{address}/api/tags'


# Helper function to format file size
def format_size(size_bytes):
    """Formats bytes into GB with one decimal place, e.g., '2.3 GB'."""
    gb = size_bytes / (1024 * 1024 * 1024)
    return f"{gb:.1f} GB"


# Helper function to format relative time
def format_relative_time(dt_str):
    """Formats an ISO 8601 datetime string into a human-readable relative time."""
    try:
        dt = datetime.datetime.fromisoformat(dt_str.replace('Z', ''))
        now = datetime.datetime.utcnow()

        diff = now - dt
        total_seconds = int(diff.total_seconds())

        if total_seconds < 0:
            return "in the future"
        elif total_seconds == 0:
            return "just now"

        time_units = [
            (31536000, "year", "years"),
            (2592000, "month", "months"),
            (604800, "week", "weeks"),
            (86400, "day", "days"),
            (3600, "hour", "hours"),
            (60, "minute", "minutes"),
            (1, "second", "seconds"),
        ]

        for seconds_in_unit, singular, plural in time_units:
            if total_seconds >= seconds_in_unit:
                count = total_seconds // seconds_in_unit
                return f"{count} {singular if count == 1 else plural} ago"
        return "just now"

    except (ValueError, TypeError):
        return "invalid date"
    except Exception:
        return "unknown time"


def list_models(args):
    try:
        response = requests.get(VKLLAMA_MODELS_URL.format(address=args.address))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        models_data = response.json().get('models', [])

        formatted_models = []
        # Initialize max lengths with header lengths
        max_name_len = len("NAME")
        max_id_len = len("ID")
        max_size_len = len("SIZE")
        max_modified_len = len("MODIFIED")

        for model in models_data:
            name = model.get('name', 'N/A')
            # Shorten digest to 12 characters, handle missing or error digests
            digest = model.get('digest')
            model_id = (digest[:12] if digest and digest != 'sha256:error_calculating_digest' else 'N/A')

            size = format_size(model.get('size', 0))
            modified = format_relative_time(model.get('modified_at', ''))

            formatted_models.append({
                'name': name,
                'id': model_id,
                'size': size,
                'modified': modified
            })

            # Update max lengths
            max_name_len = max(max_name_len, len(name))
            max_id_len = max(max_id_len, len(model_id))
            max_size_len = max(max_size_len, len(size))
            max_modified_len = max(max_modified_len, len(modified))

        # Print header with dynamic padding
        print(f"{'NAME':<{max_name_len}}  {'ID':<{max_id_len}}  {'SIZE':<{max_size_len}}  {'MODIFIED':<{max_modified_len}}")

        # Print model data with dynamic padding
        for model in formatted_models:
            print(f"{model['name']:<{max_name_len}}  {model['id']:<{max_id_len}}  {model['size']:<{max_size_len}}  {model['modified']:<{max_modified_len}}")

    except requests.exceptions.ConnectionError:
        print(f'Error: Could not connect to the vkllama server at {args.address}. Is the server running?')
    except requests.exceptions.HTTPError as e:
        print(f'Error from server ({e.response.status_code}): {e.response.text}')
    except json.JSONDecodeError:
        print('Error: Could not decode JSON response from the server. Check server logs.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
