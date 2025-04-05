import asyncio
from dotenv import load_dotenv
import os
import json
import requests
import time
from typing import Any, Optional, Dict, List
from mcp.server.fastmcp import FastMCP
from datetime import datetime

import sys

# Initialize FastMCP server
mcp = FastMCP("canvas")

# Hard-coded API key (replace with your actual API key if you don't want to use environment variables)
# If this is set, it will be used instead of looking for environment variables
HARD_CODED_CANVAS_API_KEY = "12436~kGccePMt8M2H6v7UvzrtKxYT99eKEECfA2ZDw9RxwAJQkntvH2meDByRFKeMYECe"

# Load environment variables from .env file (only used if HARD_CODED_CANVAS_API_KEY is empty)
load_dotenv()

# Simple in-memory cache
cache = {
    "courses": None,
    "modules": {},
    "module_items": {},
    "file_urls": {},
    "course_analysis": {},
    "module_analysis": {},
    "resource_analysis": {},
    "assignments": {}  # Add assignments cache
}

# Cache expiration times (in seconds)
CACHE_TTL = {
    "courses": 3600,  # 1 hour
    "modules": 1800,  # 30 minutes
    "module_items": 1800,  # 30 minutes
    "file_urls": 3600,  # 1 hour
    "course_analysis": 7200,  # 2 hours
    "module_analysis": 7200,  # 2 hours
    "resource_analysis": 7200,  # 2 hours
    "assignments": 1800  # 30 minutes
}

cache_timestamps = {
    "courses": 0,
    "modules": {},
    "module_items": {},
    "file_urls": {},
    "course_analysis": {},
    "module_analysis": {},
    "resource_analysis": {},
    "assignments": {}  # Add assignments timestamps
}

def cache_get(cache_type, key=None):
    """Get an item from cache if it exists and is not expired"""
    current_time = time.time()

    if key is None:
        # For single-item caches like courses
        if cache_type in cache and cache[cache_type] is not None:
            if current_time - cache_timestamps[cache_type] < CACHE_TTL[cache_type]:
                return cache[cache_type]
    else:
        # For multi-item caches
        if cache_type in cache and key in cache[cache_type]:
            if key in cache_timestamps[cache_type] and current_time - cache_timestamps[cache_type][key] < CACHE_TTL[cache_type]:
                return cache[cache_type][key]

    return None

def cache_set(cache_type, value, key=None):
    """Store an item in cache with current timestamp"""
    current_time = time.time()

    if key is None:
        # For single-item caches
        cache[cache_type] = value
        cache_timestamps[cache_type] = current_time
    else:
        # For multi-item caches
        if cache_type not in cache:
            cache[cache_type] = {}
        if cache_type not in cache_timestamps:
            cache_timestamps[cache_type] = {}

        cache[cache_type][key] = value
        cache_timestamps[cache_type][key] = current_time

@mcp.tool()
async def get_courses():
    """Use this tool to retrieve all available Canvas courses for the current user. This tool returns a dictionary mapping course names to their corresponding IDs. Use this when you need to find course IDs based on names, display all available courses, or when needing to access any course-related information."""
    # Check cache first
    cached_courses = cache_get("courses")
    if cached_courses:
        print("Using cached courses data")
        return cached_courses

    try:
        url = "https://canvas.kau.se/api/v1/courses?page=1&per_page=100"

        # Check if API key is available (first check hard-coded value, then environment variable)
        api_key = HARD_CODED_CANVAS_API_KEY or os.getenv('CANVAS_API_KEY')
        if not api_key:
            print("Error: No Canvas API key available. Either set HARD_CODED_CANVAS_API_KEY in the code or provide CANVAS_API_KEY environment variable")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Add timeout to prevent hanging
        response = requests.get(url, headers=headers, timeout=10)

        # Check response status
        if response.status_code != 200:
            print(f"Error: Canvas API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None

        courses = response.json()

        out = {}

        for course in courses:
            if "id" in course and "name" in course:
                out.update({
                    course["name"]: course["id"]
                })

        # Only write to file if we have data and the filesystem is writable
        if out:
            try:
                is_writable = os.access(os.getcwd(), os.W_OK)
                if is_writable:
                    with open("courses.json", "w") as f:
                        json.dump(out, f)
                    print("Saved courses data to courses.json")
            except Exception as e:
                print(f"Warning: Could not save courses to file: {e}")

            # Store in cache
            cache_set("courses", out)
        else:
            print("Warning: No courses found in Canvas API response")
            return None

        return out

    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_courses: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

@mcp.tool()
async def get_modules(course_id):
    """Use this tool to retrieve all modules within a specific Canvas course. This tool returns a list of module objects containing module details like ID, name, and status. Use this when exploring or navigating course content structure."""
    # Check cache first
    cached_modules = cache_get("modules", course_id)
    if cached_modules:
        print(f"Using cached modules data for course {course_id}")
        return cached_modules

    try:
        url = f"https://canvas.kau.se/api/v1/courses/{course_id}/modules"

        # Check if API key is available (first check hard-coded value, then environment variable)
        api_key = HARD_CODED_CANVAS_API_KEY or os.getenv('CANVAS_API_KEY')
        if not api_key:
            print("Error: No Canvas API key available. Either set HARD_CODED_CANVAS_API_KEY in the code or provide CANVAS_API_KEY environment variable")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Add timeout
        response = requests.get(url, headers=headers, timeout=10)

        # Check response status
        if response.status_code != 200:
            print(f"Error: Canvas API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None

        modules = response.json()

        # Only write to file if we have data and the filesystem is writable
        if modules:
            try:
                is_writable = os.access(os.getcwd(), os.W_OK)
                if is_writable:
                    with open("modules.json", "w") as f:
                        json.dump(modules, f)
                    print(f"Saved modules data for course {course_id} to modules.json")
            except Exception as e:
                print(f"Warning: Could not save modules to file: {e}")

            # Store in cache
            cache_set("modules", modules, course_id)
        else:
            print(f"Warning: No modules found for course {course_id}")
            return None

        return modules

    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_modules: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

@mcp.tool()
async def get_module_items(course_id, module_id):
    """Use this tool to retrieve all items within a specific module in a Canvas course. This tool returns a list of module item objects containing details like title, type, and URLs. Use this when you need to access specific learning materials, assignments, or other content within a module."""
    # Check cache first
    cache_key = f"{course_id}_{module_id}"
    cached_items = cache_get("module_items", cache_key)
    if cached_items:
        print(f"Using cached module items for module {module_id} in course {course_id}")
        return cached_items

    try:
        url = f"https://canvas.kau.se/api/v1/courses/{course_id}/modules/{module_id}/items?per_page=100"

        # Check if API key is available (first check hard-coded value, then environment variable)
        api_key = HARD_CODED_CANVAS_API_KEY or os.getenv('CANVAS_API_KEY')
        if not api_key:
            print("Error: No Canvas API key available. Either set HARD_CODED_CANVAS_API_KEY in the code or provide CANVAS_API_KEY environment variable")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Add timeout
        response = requests.get(url, headers=headers, timeout=10)

        # Check response status
        if response.status_code != 200:
            print(f"Error: Canvas API returned status code {response.status_code} for module items")
            print(f"Response: {response.text}")
            return None

        items = response.json()

        if not items:
            print(f"Warning: No items found for module {module_id} in course {course_id}")
            return None

        # Store in cache
        cache_set("module_items", items, cache_key)

        return items

    except requests.RequestException as e:
        import traceback, sys
        print(f"Error connecting to Canvas API for module items: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None
    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_module_items: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

@mcp.tool()
async def get_file_url(course_id, file_id):
    """Use this tool to get the direct download URL for a file stored in Canvas. This tool returns a URL string that can be used to access or download the file. Use this when you need direct access to file content rather than just the Canvas page URL."""
    # Check cache first
    cache_key = f"{course_id}_{file_id}"
    cached_url = cache_get("file_urls", cache_key)
    if cached_url:
        print(f"Using cached file URL for file {file_id} in course {course_id}")
        return cached_url

    try:
        url = f"https://canvas.kau.se/api/v1/courses/{course_id}/files/{file_id}"

        # Check if API key is available (first check hard-coded value, then environment variable)
        api_key = HARD_CODED_CANVAS_API_KEY or os.getenv('CANVAS_API_KEY')
        if not api_key:
            print("Error: No Canvas API key available. Either set HARD_CODED_CANVAS_API_KEY in the code or provide CANVAS_API_KEY environment variable")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Add timeout
        response = requests.get(url, headers=headers, timeout=10)

        # Check response status
        if response.status_code != 200:
            print(f"Error: Canvas API returned status code {response.status_code} for file URL")
            print(f"Response: {response.text}")
            return None

        file_data = response.json()

        if 'url' in file_data:
            # Store in cache
            cache_set("file_urls", file_data['url'], cache_key)
            return file_data['url']

        print(f"Warning: No URL found in file data for file {file_id}")
        return None

    except requests.RequestException as e:
        import traceback, sys
        print(f"Error connecting to Canvas API for file URL: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None
    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_file_url: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

def analyze_image_with_gemini(image_path):
    """
    Simplified version that doesn't use Gemini
    Always returns None as image analysis is no longer supported
    """
    print(f"Image analysis is no longer supported. Ignoring image: {image_path}")
    return None

def analyze_query_with_gemini(query, courses, course_modules=None):
    """
    Simplified version that uses keyword matching instead of Gemini
    Returns a dict with course_id, module_ids, and reasoning
    """
    # Check cache first
    if course_modules is None:
        # First stage: identify the correct course
        cache_key = query.lower()[:100]  # Limit key size
        cached_analysis = cache_get("course_analysis", cache_key)
        if cached_analysis:
            print(f"Using cached course analysis for query: {query[:30]}...")
            return cached_analysis
    else:
        # Second stage: identify relevant modules
        course_id = next(iter(courses.values()))  # Get any course ID to use in cache key
        cache_key = f"{course_id}_{query.lower()[:100]}"  # Limit key size
        cached_analysis = cache_get("module_analysis", cache_key)
        if cached_analysis:
            print(f"Using cached module analysis for query: {query[:30]}...")
            return cached_analysis

    try:
        if course_modules is None:
            # First stage: identify the correct course
            # Extract keywords from the query (words longer than 3 characters)
            query_keywords = [word.lower() for word in query.split() if len(word) > 3]

            # Find the best matching course based on keyword overlap
            best_match = None
            best_score = 0
            best_overlap = 0

            for name, id in courses.items():
                # Extract keywords from the course name
                course_keywords = [word.lower() for word in name.split() if len(word) > 3]

                # Calculate overlap score
                overlap = sum(1 for keyword in query_keywords if any(keyword in course_word for course_word in course_keywords))
                if overlap > 0:
                    score = overlap / len(query_keywords) if query_keywords else 0

                    if score > best_score or (score == best_score and overlap > best_overlap):
                        best_score = score
                        best_overlap = overlap
                        best_match = name

            # If no good match found, use the first course as fallback
            if not best_match and courses:
                best_match = next(iter(courses.keys()))
                best_score = 0.1

            if best_match:
                result = {
                    "course_name": best_match,
                    "confidence": best_score,
                    "reasoning": f"Found {best_overlap} keyword matches between query and course name"
                }

                # Cache the result
                cache_set("course_analysis", result, cache_key)
                return result
            else:
                return None
        else:
            # Second stage: identify relevant modules
            # Extract keywords from the query (words longer than 3 characters)
            query_keywords = [word.lower() for word in query.split() if len(word) > 3]

            # Find modules with matching keywords
            relevant_modules = []
            relevance_explanations = {}

            for module in course_modules:
                module_name = module["name"]
                module_keywords = [word.lower() for word in module_name.split() if len(word) > 3]

                # Calculate overlap score
                overlap = sum(1 for keyword in query_keywords if any(keyword in module_word for module_word in module_keywords))
                if overlap > 0:
                    relevant_modules.append(module_name)
                    relevance_explanations[module_name] = f"Found {overlap} keyword matches"

            # Limit to top 3 modules
            if len(relevant_modules) > 3:
                relevant_modules = relevant_modules[:3]

            # If no relevant modules found, include the first module as fallback
            if not relevant_modules and course_modules:
                relevant_modules = [course_modules[0]["name"]]
                relevance_explanations[course_modules[0]["name"]] = "Default module (no keyword matches found)"

            result = {
                "module_names": relevant_modules,
                "relevance_explanations": relevance_explanations
            }

            # Cache the result
            course_id = next(iter(courses.values()))  # Get any course ID to use in cache key
            cache_key = f"{course_id}_{query.lower()[:100]}"
            cache_set("module_analysis", result, cache_key)

            return result
    except Exception as e:
        import traceback, sys
        print(f"Error in analyze_query_with_gemini: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

def analyze_resource_relevance(query, resource_items, course_name, module_name):
    """
    Simplified version that uses keyword matching instead of Gemini
    Returns a dict with resource_indices, relevance_scores, and reasoning
    """
    # Check cache first
    cache_key = f"{course_name}_{module_name}_{query.lower()[:100]}"
    cached_analysis = cache_get("resource_analysis", cache_key)
    if cached_analysis:
        print(f"Using cached resource analysis for module {module_name}")
        return cached_analysis

    try:
        # Extract keywords from the query (words longer than 3 characters)
        query_keywords = [word.lower() for word in query.split() if len(word) > 3]

        # If no keywords found, use all words
        if not query_keywords:
            query_keywords = [word.lower() for word in query.split()]

        # Calculate relevance scores for each resource
        resource_scores = []

        for idx, item in enumerate(resource_items):
            title = item.get('title', '').lower()
            item_type = item.get('type', '').lower()

            # Skip certain types of items that are less likely to be content
            if item_type in ['ExternalUrl', 'SubHeader']:
                continue

            # Calculate keyword matches in title
            title_words = [word.lower() for word in title.split()]
            matches = sum(1 for keyword in query_keywords if any(keyword in word for word in title_words))

            # Assign a base score based on item type
            type_score = 0.5
            if item_type == 'File':
                type_score = 0.8  # Files are usually more relevant
            elif item_type == 'Assignment':
                type_score = 0.7  # Assignments are also relevant
            elif item_type == 'Page':
                type_score = 0.6  # Pages are somewhat relevant

            # Calculate final score
            if matches > 0:
                # More matches = higher score
                score = min(0.9, 0.3 + (matches * 0.2) + type_score * 0.5)
            else:
                # No matches = lower score based on type
                score = type_score * 0.3

            resource_scores.append((idx, score))

        # Sort by score (descending) and take top 5
        resource_scores.sort(key=lambda x: x[1], reverse=True)
        top_resources = resource_scores[:5]

        # If no resources found, return None
        if not top_resources:
            return None

        # Format the result
        result = {
            "resource_indices": [idx for idx, _ in top_resources],
            "relevance_scores": [score for _, score in top_resources],
            "reasoning": f"Found resources matching {len(query_keywords)} keywords from query"
        }

        # Cache the result
        cache_set("resource_analysis", result, cache_key)

        return result

    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in analyze_resource_relevance: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

@mcp.tool()
async def find_resources(query: str, image_path: Optional[str] = None):
    """Use this tool to search for and identify the most relevant learning resources across Canvas courses based on a text query or image. This tool analyzes user needs and returns resources ranked by relevance. Use this when helping users find specific learning materials, lecture notes, or content related to their questions."""
    try:
        print(f"Processing query: {query}, image_path: {image_path}")

        # Validate inputs
        if not query and not image_path:
            return {"error": "Either query or image_path must be provided"}

        # Check if image path exists when provided
        if image_path and not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}")
            if not query:
                return {"error": "Image path provided does not exist and no query was provided"}

        # Call helper_resources with proper error handling
        result = await helper_resources(query, image_path)
        if not result:
            return {"error": "Failed to find resources", "message": "Resource search returned no results"}

        # Check if result indicates an error
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            return result[0]

        return result

    except Exception as e:
        # Log the error
        import traceback, sys
        error_msg = f"Error processing request: {str(e)}"
        line_num = sys.exc_info()[-1].tb_lineno
        print(f"{error_msg} at line {line_num}")
        print(traceback.format_exc())

        # Return a proper error response
        return {"error": "Internal server error", "message": str(e), "line": line_num}

async def process_module(course_id, module, query, course_name):
    """Process a single module and return relevant resources"""
    module_id = module["id"]
    module_name = module["name"]

    print(f"Getting items for module: {module_name} (ID: {module_id})")

    # Get all items in this module
    items = await get_module_items(course_id, module_id)

    if not items:
        print(f"No items found in module: {module_name}")
        return []

    print(f"Found {len(items)} items in module: {module_name}")

    # Analyze which resources are most relevant within this module
    resource_analysis = analyze_resource_relevance(query, items, course_name, module_name)
    if not resource_analysis:
        print(f"Failed to analyze resource relevance for module: {module_name}")
        return []

    relevant_indices = resource_analysis.get("resource_indices", [])
    relevance_scores = resource_analysis.get("relevance_scores", [])

    if not relevant_indices:
        print(f"No relevant resources identified in module: {module_name}")
        return []

    print(f"Relevant indices from analysis: {relevant_indices}")

    # Add the relevant resources to our results
    relevant_resources = []
    file_url_tasks = []

    for idx_pos, idx in enumerate(relevant_indices):
        if idx < len(items):
            item = items[idx]
            # Format the resource for the response
            resource = {
                "title": item.get("title", "Untitled Resource"),
                "type": item.get("type", "Unknown"),
                "url": item.get("html_url", ""),
                "course": course_name,
                "module": module_name,
                "relevance_score": relevance_scores[idx_pos] if idx_pos < len(relevance_scores) else 0.5,
                "item": item  # Save the original item for file URL resolution
            }

            relevant_resources.append(resource)

    # Process file URLs in parallel
    for resource in relevant_resources:
        item = resource.pop("item")  # Remove the item from the resource
        if item.get("type") == "File" and "content_id" in item:
            file_url = await get_file_url(course_id, item["content_id"])
            if file_url:
                resource["url"] = file_url

    return relevant_resources

async def helper_resources(query, image_path=None):
    """
    Main function to find resources based on a student query or image
    Returns a list of relevant resources
    """
    # If an image is provided, analyze it to get a query
    if image_path and os.path.exists(image_path):
        print(f"Analyzing image: {image_path}")
        image_query = analyze_image_with_gemini(image_path)
        # Combine the original query with the image analysis if image analysis was successful
        if image_query and query:
            query = f"{query} - {image_query}"
        elif image_query:
            query = image_query
        print(f"Image analysis result: {image_query if image_query else 'Failed to analyze image'}")

    # Step 1: Get all courses
    courses = await get_courses()
    if not courses:
        print("Error: No courses found from Canvas API")
        return [{"error": "No courses found", "details": "Please check your Canvas API configuration"}]

    # Print available courses for debugging
    print(f"Available courses: {list(courses.keys())}")

    # Step 2: Analyze the query to identify the relevant course
    print(f"Analyzing query: '{query}'")
    course_analysis = analyze_query_with_gemini(query, courses)
    if not course_analysis:
        print("Error: Failed to analyze query to determine relevant course")
        return [{"error": "Failed to analyze query", "details": "Could not determine relevant course"}]

    course_name = course_analysis.get("course_name")
    confidence = course_analysis.get("confidence", 0)
    reasoning = course_analysis.get("reasoning", "No reasoning provided")

    print(f"Course analysis result: {course_name} (confidence: {confidence})")
    print(f"Reasoning: {reasoning}")

    # Improved course matching logic
    if course_name not in courses:
        print(f"Exact course match not found for '{course_name}', trying partial matches")
        # Try to find a partial match with higher threshold
        best_match = None
        best_match_score = 0

        for name in courses:
            # Check for significant word overlap
            query_words = set(w.lower() for w in query.split() if len(w) > 3)
            name_words = set(w.lower() for w in name.split() if len(w) > 3)

            # Calculate overlap score
            if query_words and name_words:
                overlap = len(query_words.intersection(name_words))
                score = overlap / min(len(query_words), len(name_words))
                print(f"Course: {name}, score: {score}")

                if score > best_match_score:
                    best_match_score = score
                    best_match = name

            # Also check for simple substring match
            elif course_name.lower() in name.lower() or name.lower() in course_name.lower():
                print(f"Substring match found: {name}")
                if best_match is None:
                    best_match = name

        if best_match and best_match_score >= 0.2:
            print(f"Using best match course: {best_match} (score: {best_match_score})")
            course_name = best_match
        elif best_match:
            print(f"Using substring match course: {best_match}")
            course_name = best_match
        else:
            # If still no match found, check if there's a course related to the query topic
            query_keywords = [w.lower() for w in query.split() if len(w) > 4]

            for name in courses:
                for keyword in query_keywords:
                    if keyword in name.lower():
                        print(f"Keyword match found: {name} (keyword: {keyword})")
                        course_name = name
                        break
                if course_name in courses:
                    break

            # Last resort - default to first course
            if course_name not in courses:
                print("No course match found and couldn't determine a relevant course")
                return [{"error": "No relevant course found", "query": query}]

    course_id = courses[course_name]
    print(f"Selected course: {course_name} (ID: {course_id})")

    # Step 3: Get modules for the identified course
    modules = await get_modules(course_id)
    if not modules:
        print(f"No modules found for course: {course_name}")
        return [{"error": "No modules found", "course": course_name}]

    print(f"Found {len(modules)} modules in course {course_name}")

    # Step 4: Analyze which modules are most relevant
    module_analysis = analyze_query_with_gemini(query, courses, modules)
    if not module_analysis:
        print("Error: Failed to analyze query to determine relevant modules")
        return [{"error": "Failed to analyze query", "details": "Could not determine relevant modules"}]

    relevant_module_names = module_analysis.get("module_names", [])

    print(f"Relevant module names from analysis: {relevant_module_names}")

    # Map module names to IDs
    relevant_modules = []
    for module in modules:
        if module["name"] in relevant_module_names:
            print(f"Found exact module match: {module['name']}")
            relevant_modules.append(module)
        # If we can't find exact matches, include modules that contain the query keywords
        elif any(keyword.lower() in module["name"].lower() 
                for keyword in query.lower().split() if len(keyword) > 3):
            print(f"Found keyword match in module: {module['name']}")
            relevant_modules.append(module)

    # If no relevant modules found
    if not relevant_modules:
        print("No relevant modules found for the query")
        return [{"error": "No relevant modules found", "course": course_name, "query": query}]

    print(f"Selected {len(relevant_modules)} modules for further analysis")

    # Step 5: Process modules in parallel
    module_tasks = [process_module(course_id, module, query, course_name) for module in relevant_modules]
    module_results = await asyncio.gather(*module_tasks)

    # Combine results from all modules
    all_relevant_resources = []
    for result in module_results:
        all_relevant_resources.extend(result)

    # Sort resources by relevance score (descending)
    if all_relevant_resources:
        all_relevant_resources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Return results, or an error if none found
    if all_relevant_resources:
        print(f"Returning {len(all_relevant_resources)} relevant resources")
        # Try to save resources to a json file, but don't fail if it's not possible (e.g., read-only filesystem)
        try:
            # Check if we're in a writable environment before attempting to write
            is_writable = os.access(os.getcwd(), os.W_OK)
            if is_writable:
                safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]  # Make filename safe
                with open(f"resources_{safe_query}.json", "w") as f:
                    json.dump(all_relevant_resources, f)
                print(f"Saved resources to resources_{safe_query}.json")
        except Exception as e:
            # Just log the error but continue
            print(f"Warning: Could not save resources to file: {e}")

        return all_relevant_resources
    else:
        print(f"No relevant resources found for query: {query}")
        return [{"error": "No relevant resources found", "query": query, "course": course_name}]

@mcp.tool()
async def get_course_assignments(course_id: str, bucket: str = None):
    """Use this tool to retrieve all assignments for a specific Canvas course, with optional filtering by status. This tool returns assignment details including name, description, due date, and submission status. Use this when helping users manage their coursework, check due dates, or find assignment details.

    Args:
        course_id: The Canvas course ID
        bucket: Optional filter - past, overdue, undated, ungraded, unsubmitted, upcoming, future
    """

    try:
        # Build URL with optional bucket parameter
        url = f"https://canvas.kau.se/api/v1/courses/{course_id}/assignments"
        params = {
            "order_by": "due_at",
            "per_page": 100,  # Get max assignments per page
            "include[]": ["submission", "all_dates"]  # Include submission details
        }
        if bucket:
            params["bucket"] = bucket

        # Check if API key is available (first check hard-coded value, then environment variable)
        api_key = HARD_CODED_CANVAS_API_KEY or os.getenv('CANVAS_API_KEY')
        if not api_key:
            print("Error: No Canvas API key available. Either set HARD_CODED_CANVAS_API_KEY in the code or provide CANVAS_API_KEY environment variable")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Add timeout to prevent hanging
        response = requests.get(url, headers=headers, params=params, timeout=10)

        # Check response status
        if response.status_code != 200:
            print(f"Error: Canvas API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None

        assignments = response.json()

        # Store in cache
        # cache_set("assignments", assignments, cache_key)

        # only return "id", "description", "due_at", "has_submitted_submissions" and "name" of the assignments
        return [{"id": assignment["id"], "description": assignment["description"], "due_at": assignment["due_at"], "has_submitted_submissions": assignment["has_submitted_submissions"], "name": assignment["name"]} for assignment in assignments]


    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_course_assignments: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

@mcp.tool()
async def get_assignments_by_course_name(course_name: str, bucket: str = None):
    """Use this tool to retrieve all assignments for a Canvas course using its name rather than ID. This tool returns assignment details the same as get_course_assignments. Use this when you have the course name but not the ID, or when helping users find assignments across multiple courses.

    Args:
        course_name: The name of the course as it appears in Canvas (partial matches supported)
        bucket: Optional filter - past, overdue, undated, ungraded, unsubmitted, upcoming, future
    """
    try:
        # First get all courses to find the course ID
        courses = await get_courses()
        if not courses:
            print("Error: Could not fetch courses")
            return None

        course_found = False

        for courseName, courseId in courses.items():
            if course_name in courseName:
                course_id = courseId
                course_found = True
                break

        # Find the course ID by name
        if not course_found:
            print(f"Error: Course '{course_name}' not found")
            print(f"Available courses: {list(courses.keys())}")
            return None

        # Get assignments using the course ID
        return await get_course_assignments(course_id, bucket)

    except Exception as e:
        import traceback, sys
        print(f"Unexpected error in get_assignments_by_course_name: {e}")
        print(f"Line number: {sys.exc_info()[-1].tb_lineno}")
        print(traceback.format_exc())
        return None

# @mcp.tool()
# async def get_gradescope_course_by_name(course_name: str):
#     """Get a course from Gradescope by name"""
#     courses = await get_gradescope_courses()
#     for course in courses["student"].values():
#         if course_name in course["name"] or course_name in course["full_name"]:
#             return course
#     return None

# Add test functions
async def test_assignments():
    """Test the assignment functions"""
    print("\nTesting assignment functions...")

    # Test get_course_assignments
    print("\nTesting get_course_assignments...")
    courses = await get_courses()
    if courses:
        first_course_id = next(iter(courses.values()))
        print(f"Testing with first course ID: {first_course_id}")

        # Test without bucket
        assignments = await get_course_assignments(first_course_id)
        print(f"Found {len(assignments) if assignments else 0} assignments without bucket")

        # Test with bucket
        upcoming = await get_course_assignments(first_course_id, "upcoming")
        print(f"Found {len(upcoming) if upcoming else 0} upcoming assignments")

    # Test get_assignments_by_course_name
    print("\nTesting get_assignments_by_course_name...")
    if courses:
        first_course_name = next(iter(courses.keys()))
        print(f"Testing with first course name: {first_course_name}")

        # Test without bucket
        assignments = await get_assignments_by_course_name(first_course_name)
        print(f"Found {len(assignments) if assignments else 0} assignments without bucket")

        # Test with bucket
        upcoming = await get_assignments_by_course_name(first_course_name, "upcoming")
        print(f"Found {len(upcoming) if upcoming else 0} upcoming assignments")

# Add MCP tools for Gradescope

@mcp.tool()
async def get_canvas_courses():
    """Use this tool to retrieve all available Canvas courses for the current user. This is an alias for get_courses. Use this when you need to find course IDs based on names or display all available courses."""
    return await get_courses()


@mcp.tool()
async def search_education_platforms(query: str):
    """Use this tool to search for information across Canvas using natural language queries. This tool analyzes the query and returns appropriately formatted results. Use this for educational queries when you need to find relevant resources.

    Args:
        query: Natural language query about courses, assignments, or other educational content
    """
    # Handle general queries about academics or courses
    if "course" in query.lower():
        # Get Canvas courses
        try:
            canvas_courses = await get_courses()
            if canvas_courses:
                return {
                    "message": "Here are your Canvas courses:",
                    "results": {"canvas": {"courses": canvas_courses}}
                }
        except Exception as e:
            print(f"Error fetching Canvas courses: {e}")
            return {"error": f"Error fetching Canvas courses: {str(e)}"}

    # Use the Canvas resource finder
    try:
        resources = await find_resources(query=query)
        return {
            "message": "Here are the most relevant Canvas resources for your query:",
            "source": "Canvas",
            "resources": resources
        }
    except Exception as e:
        print(f"Error searching Canvas: {e}")
        return {"error": f"Error searching Canvas: {str(e)}"}

# Add a test for the unified search function
async def test_unified_search():
    """Test the unified search function"""
    print("\nTesting unified search...")

    # Test a Canvas-specific query
    print("\nTesting Canvas-specific query...")
    canvas_result = await search_education_platforms("What resources are available for learning matrices in Canvas?")
    print(f"Canvas search result: {canvas_result is not None}")

    # Test a general query
    print("\nTesting general query...")
    general_result = await search_education_platforms("What courses am I enrolled in?")
    print(f"General search result: {general_result is not None}")

# Update run_tests to include the unified search test
async def run_tests():
    # Existing tests
    out = await find_resources(query="what would be the best resources to learn dot product of matrices from canvas?")
    print(out)

    print("="*50)

    # Add assignment tests
    await test_assignments()

    print("="*50)

    # Get assignments for Linear Algebra
    assignments = await get_assignments_by_course_name("Linear Algebra")
    print(assignments)

    with open("assignments.json", "w") as f:
        json.dump(assignments, f)

    print("="*50)

    # Add unified search test
    await test_unified_search()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Initialize and run the server
        mcp.run(transport='stdio')
    else:
        # Run tests
        asyncio.run(run_tests())
