import asyncio
from dotenv import load_dotenv
import os
import json
import requests
import time
from typing import Any, Optional, Dict, List, Union
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("canvas_mcp")

# Initialize FastMCP server
mcp = FastMCP("canvas")

# Load environment variables from .env file
load_dotenv()

# Configuration
class Config:
    """Configuration for Canvas MCP server"""
    # Canvas API configuration
    CANVAS_API_KEY = os.getenv('CANVAS_API_KEY')
    CANVAS_BASE_URL = os.getenv('CANVAS_BASE_URL', 'https://canvas.kau.se')

    # Cache configuration
    CACHE_TTL = {
        "courses": 3600,        # 1 hour
        "modules": 1800,        # 30 minutes
        "module_items": 1800,   # 30 minutes
        "file_urls": 3600,      # 1 hour
        "course_analysis": 7200,  # 2 hours
        "module_analysis": 7200,  # 2 hours
        "resource_analysis": 7200,  # 2 hours
        "assignments": 1800     # 30 minutes
    }

    # API request configuration
    REQUEST_TIMEOUT = 10  # seconds

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get the Canvas API key from environment variables"""
        if not cls.CANVAS_API_KEY:
            logger.error("No Canvas API key available. Please set CANVAS_API_KEY environment variable")
            return None
        return cls.CANVAS_API_KEY

    @classmethod
    def get_api_url(cls, endpoint: str) -> str:
        """Get the full URL for a Canvas API endpoint"""
        return f"{cls.CANVAS_BASE_URL}/api/v1/{endpoint}"

# Cache implementation
class Cache:
    """Simple in-memory cache with TTL"""
    _data = {
        "courses": None,
        "modules": {},
        "module_items": {},
        "file_urls": {},
        "course_analysis": {},
        "module_analysis": {},
        "resource_analysis": {},
        "assignments": {}
    }

    _timestamps = {
        "courses": 0,
        "modules": {},
        "module_items": {},
        "file_urls": {},
        "course_analysis": {},
        "module_analysis": {},
        "resource_analysis": {},
        "assignments": {}
    }

    @classmethod
    def get(cls, cache_type: str, key: Optional[str] = None) -> Any:
        """Get an item from cache if it exists and is not expired

        Args:
            cache_type: Type of cache (courses, modules, etc.)
            key: Optional key for multi-item caches

        Returns:
            Cached value or None if not found or expired
        """
        current_time = time.time()

        if key is None:
            # For single-item caches like courses
            if cache_type in cls._data and cls._data[cache_type] is not None:
                if current_time - cls._timestamps[cache_type] < Config.CACHE_TTL[cache_type]:
                    logger.debug(f"Cache hit for {cache_type}")
                    return cls._data[cache_type]
        else:
            # For multi-item caches
            if cache_type in cls._data and key in cls._data[cache_type]:
                if key in cls._timestamps[cache_type] and current_time - cls._timestamps[cache_type][key] < Config.CACHE_TTL[cache_type]:
                    logger.debug(f"Cache hit for {cache_type}/{key}")
                    return cls._data[cache_type][key]

        logger.debug(f"Cache miss for {cache_type}/{key if key else ''}")
        return None

    @classmethod
    def set(cls, cache_type: str, value: Any, key: Optional[str] = None) -> None:
        """Store an item in cache with current timestamp

        Args:
            cache_type: Type of cache (courses, modules, etc.)
            value: Value to store
            key: Optional key for multi-item caches
        """
        current_time = time.time()

        if key is None:
            # For single-item caches
            cls._data[cache_type] = value
            cls._timestamps[cache_type] = current_time
        else:
            # For multi-item caches
            if cache_type not in cls._data:
                cls._data[cache_type] = {}
            if cache_type not in cls._timestamps:
                cls._timestamps[cache_type] = {}

            cls._data[cache_type][key] = value
            cls._timestamps[cache_type][key] = current_time

        logger.debug(f"Cached {cache_type}/{key if key else ''}")

@mcp.tool()
async def get_courses():
    """Use this tool to retrieve all available Canvas courses for the current user.

    This tool returns a dictionary mapping course names to their corresponding IDs.
    Use this when you need to find course IDs based on names, display all available 
    courses, or when needing to access any course-related information.

    Returns:
        Dict[str, int]: Dictionary mapping course names to their IDs, or None if an error occurs
    """
    # Check cache first
    cached_courses = Cache.get("courses")
    if cached_courses:
        logger.info("Using cached courses data")
        return cached_courses

    try:
        # Get API key
        api_key = Config.get_api_key()
        if not api_key:
            return None

        # Construct API URL
        url = Config.get_api_url("courses")

        # Set up request parameters
        params = {
            "page": 1,
            "per_page": 100
        }

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Make API request with timeout
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Check response status
        if response.status_code != 200:
            logger.error(f"Canvas API returned status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None

        # Process response
        courses = response.json()
        course_dict = {}

        for course in courses:
            if "id" in course and "name" in course:
                course_dict[course["name"]] = course["id"]

        # Store in cache if we have data
        if course_dict:
            Cache.set("courses", course_dict)
            logger.info(f"Retrieved {len(course_dict)} courses from Canvas API")
        else:
            logger.warning("No courses found in Canvas API response")
            return None

        return course_dict

    except requests.RequestException as e:
        logger.error(f"Error connecting to Canvas API: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_courses: {e}")
        return None

@mcp.tool()
async def get_modules(course_id: Union[int, str]):
    """Use this tool to retrieve all modules within a specific Canvas course.

    This tool returns a list of module objects containing module details like ID, name, 
    and status. Use this when exploring or navigating course content structure.

    Args:
        course_id: The Canvas course ID

    Returns:
        List[Dict]: List of module objects, or None if an error occurs
    """
    # Check cache first
    cached_modules = Cache.get("modules", str(course_id))
    if cached_modules:
        logger.info(f"Using cached modules data for course {course_id}")
        return cached_modules

    try:
        # Get API key
        api_key = Config.get_api_key()
        if not api_key:
            return None

        # Construct API URL
        url = Config.get_api_url(f"courses/{course_id}/modules")

        # Set up request parameters
        params = {
            "per_page": 100
        }

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Make API request with timeout
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Check response status
        if response.status_code != 200:
            logger.error(f"Canvas API returned status code {response.status_code} for modules")
            logger.error(f"Response: {response.text}")
            return None

        # Process response
        modules = response.json()

        # Store in cache if we have data
        if modules:
            Cache.set("modules", modules, str(course_id))
            logger.info(f"Retrieved {len(modules)} modules for course {course_id}")
        else:
            logger.warning(f"No modules found for course {course_id}")
            return None

        return modules

    except requests.RequestException as e:
        logger.error(f"Error connecting to Canvas API for modules: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_modules: {e}")
        return None

@mcp.tool()
async def get_module_items(course_id: Union[int, str], module_id: Union[int, str]):
    """Use this tool to retrieve all items within a specific module in a Canvas course.

    This tool returns a list of module item objects containing details like title, type, 
    and URLs. Use this when you need to access specific learning materials, assignments, 
    or other content within a module.

    Args:
        course_id: The Canvas course ID
        module_id: The Canvas module ID

    Returns:
        List[Dict]: List of module item objects, or None if an error occurs
    """
    # Check cache first
    cache_key = f"{course_id}_{module_id}"
    cached_items = Cache.get("module_items", cache_key)
    if cached_items:
        logger.info(f"Using cached module items for module {module_id} in course {course_id}")
        return cached_items

    try:
        # Get API key
        api_key = Config.get_api_key()
        if not api_key:
            return None

        # Construct API URL
        url = Config.get_api_url(f"courses/{course_id}/modules/{module_id}/items")

        # Set up request parameters
        params = {
            "per_page": 100
        }

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Make API request with timeout
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Check response status
        if response.status_code != 200:
            logger.error(f"Canvas API returned status code {response.status_code} for module items")
            logger.error(f"Response: {response.text}")
            return None

        # Process response
        items = response.json()

        # Store in cache if we have data
        if items:
            Cache.set("module_items", items, cache_key)
            logger.info(f"Retrieved {len(items)} items for module {module_id} in course {course_id}")
        else:
            logger.warning(f"No items found for module {module_id} in course {course_id}")
            return None

        return items

    except requests.RequestException as e:
        logger.error(f"Error connecting to Canvas API for module items: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_module_items: {e}")
        return None

@mcp.tool()
async def get_file_url(course_id: Union[int, str], file_id: Union[int, str]):
    """Use this tool to get the direct download URL for a file stored in Canvas.

    This tool returns a URL string that can be used to access or download the file. 
    Use this when you need direct access to file content rather than just the Canvas page URL.

    Args:
        course_id: The Canvas course ID
        file_id: The Canvas file ID

    Returns:
        str: Direct download URL for the file, or None if an error occurs
    """
    # Check cache first
    cache_key = f"{course_id}_{file_id}"
    cached_url = Cache.get("file_urls", cache_key)
    if cached_url:
        logger.info(f"Using cached file URL for file {file_id} in course {course_id}")
        return cached_url

    try:
        # Get API key
        api_key = Config.get_api_key()
        if not api_key:
            return None

        # Construct API URL
        url = Config.get_api_url(f"courses/{course_id}/files/{file_id}")

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Make API request with timeout
        response = requests.get(
            url, 
            headers=headers, 
            timeout=Config.REQUEST_TIMEOUT
        )

        # Check response status
        if response.status_code != 200:
            logger.error(f"Canvas API returned status code {response.status_code} for file URL")
            logger.error(f"Response: {response.text}")
            return None

        # Process response
        file_data = response.json()

        if 'url' in file_data:
            # Store in cache
            Cache.set("file_urls", file_data['url'], cache_key)
            logger.info(f"Retrieved file URL for file {file_id} in course {course_id}")
            return file_data['url']

        logger.warning(f"No URL found in file data for file {file_id}")
        return None

    except requests.RequestException as e:
        logger.error(f"Error connecting to Canvas API for file URL: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_file_url: {e}")
        return None

def analyze_image(image_path: str) -> None:
    """Analyze an image to extract text or content (not supported).

    This function is a placeholder that indicates image analysis is not supported.
    It always returns None and logs a warning message.

    Args:
        image_path: Path to the image file

    Returns:
        None: Image analysis is not supported
    """
    logger.warning(f"Image analysis is not supported. Ignoring image: {image_path}")
    return None

def analyze_query(query: str, courses: Dict[str, Union[int, str]], course_modules: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Analyze a query to identify relevant courses or modules using keyword matching.

    This function uses keyword matching to determine which course or module is most relevant
    to the user's query. It operates in two stages:
    1. If course_modules is None, it identifies the most relevant course
    2. If course_modules is provided, it identifies the most relevant modules within that course

    Args:
        query: The user's query text
        courses: Dictionary mapping course names to their IDs
        course_modules: Optional list of module objects for a specific course

    Returns:
        Dict: Analysis result containing course or module information, or None if analysis fails
    """
    # Check cache first
    if course_modules is None:
        # First stage: identify the correct course
        cache_key = query.lower()[:100]  # Limit key size
        cached_analysis = Cache.get("course_analysis", cache_key)
        if cached_analysis:
            logger.info(f"Using cached course analysis for query: {query[:30]}...")
            return cached_analysis
    else:
        # Second stage: identify relevant modules
        course_id = next(iter(courses.values()))  # Get any course ID to use in cache key
        cache_key = f"{course_id}_{query.lower()[:100]}"  # Limit key size
        cached_analysis = Cache.get("module_analysis", cache_key)
        if cached_analysis:
            logger.info(f"Using cached module analysis for query: {query[:30]}...")
            return cached_analysis

    try:
        if course_modules is None:
            # First stage: identify the correct course
            return _analyze_query_for_course(query, courses, cache_key)
        else:
            # Second stage: identify relevant modules
            return _analyze_query_for_modules(query, courses, course_modules, cache_key)
    except Exception as e:
        logger.exception(f"Error in analyze_query: {e}")
        return None

def _analyze_query_for_course(query: str, courses: Dict[str, Union[int, str]], cache_key: str) -> Optional[Dict]:
    """Helper function to analyze query for course relevance.

    Args:
        query: The user's query text
        courses: Dictionary mapping course names to their IDs
        cache_key: Cache key for storing results

    Returns:
        Dict: Analysis result containing course information, or None if analysis fails
    """
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
        Cache.set("course_analysis", result, cache_key)
        logger.info(f"Analyzed query for course relevance: {best_match} (confidence: {best_score:.2f})")
        return result
    else:
        logger.warning("No matching course found for query")
        return None

def _analyze_query_for_modules(query: str, courses: Dict[str, Union[int, str]], 
                              course_modules: List[Dict], cache_key: str) -> Dict:
    """Helper function to analyze query for module relevance.

    Args:
        query: The user's query text
        courses: Dictionary mapping course names to their IDs
        course_modules: List of module objects for a specific course
        cache_key: Cache key for storing results

    Returns:
        Dict: Analysis result containing module information
    """
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
    Cache.set("module_analysis", result, cache_key)
    logger.info(f"Analyzed query for module relevance: found {len(relevant_modules)} relevant modules")

    return result

def analyze_resource_relevance(query: str, resource_items: List[Dict], 
                           course_name: str, module_name: str) -> Optional[Dict]:
    """Analyze the relevance of resources to a query using keyword matching.

    This function calculates relevance scores for each resource based on keyword matches
    and resource type, then returns the top 5 most relevant resources.

    Args:
        query: The user's query text
        resource_items: List of resource items to analyze
        course_name: Name of the course containing the resources
        module_name: Name of the module containing the resources

    Returns:
        Dict: Analysis result containing resource indices, relevance scores, and reasoning,
              or None if analysis fails
    """
    # Check cache first
    cache_key = f"{course_name}_{module_name}_{query.lower()[:100]}"
    cached_analysis = Cache.get("resource_analysis", cache_key)
    if cached_analysis:
        logger.info(f"Using cached resource analysis for module {module_name}")
        return cached_analysis

    try:
        # Extract keywords from the query (words longer than 3 characters)
        query_keywords = [word.lower() for word in query.split() if len(word) > 3]

        # If no keywords found, use all words
        if not query_keywords:
            query_keywords = [word.lower() for word in query.split()]
            logger.debug("No long keywords found, using all words from query")

        logger.debug(f"Analyzing {len(resource_items)} resources with {len(query_keywords)} keywords")

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
            type_score = _get_resource_type_score(item_type)

            # Calculate final score
            if matches > 0:
                # More matches = higher score
                score = min(0.9, 0.3 + (matches * 0.2) + type_score * 0.5)
            else:
                # No matches = lower score based on type
                score = type_score * 0.3

            resource_scores.append((idx, score))
            logger.debug(f"Resource '{title}' ({item_type}) score: {score:.2f} ({matches} matches)")

        # Sort by score (descending) and take top 5
        resource_scores.sort(key=lambda x: x[1], reverse=True)
        top_resources = resource_scores[:5]

        # If no resources found, return None
        if not top_resources:
            logger.warning(f"No relevant resources found for query in module {module_name}")
            return None

        # Format the result
        result = {
            "resource_indices": [idx for idx, _ in top_resources],
            "relevance_scores": [score for _, score in top_resources],
            "reasoning": f"Found resources matching {len(query_keywords)} keywords from query"
        }

        # Cache the result
        Cache.set("resource_analysis", result, cache_key)
        logger.info(f"Found {len(top_resources)} relevant resources in module {module_name}")

        return result

    except Exception as e:
        logger.exception(f"Unexpected error in analyze_resource_relevance: {e}")
        return None

def _get_resource_type_score(item_type: str) -> float:
    """Get a base relevance score for a resource based on its type.

    Args:
        item_type: The type of the resource item

    Returns:
        float: Base relevance score between 0.0 and 1.0
    """
    # Assign scores based on item type
    if item_type == 'File':
        return 0.8  # Files are usually more relevant
    elif item_type == 'Assignment':
        return 0.7  # Assignments are also relevant
    elif item_type == 'Page':
        return 0.6  # Pages are somewhat relevant
    elif item_type == 'Discussion':
        return 0.6  # Discussions can be relevant
    elif item_type == 'Quiz':
        return 0.7  # Quizzes are relevant for learning
    else:
        return 0.5  # Default score for other types

@mcp.tool()
async def find_resources(query: str, image_path: Optional[str] = None):
    """Use this tool to search for and identify the most relevant learning resources across Canvas courses.

    This tool analyzes the user's query and returns resources ranked by relevance. It can help
    find specific learning materials, lecture notes, or content related to questions.

    Args:
        query: Text query describing the resources to find
        image_path: Optional path to an image file (not supported in current version)

    Returns:
        List[Dict]: List of relevant resources with details, or error information
    """
    try:
        logger.info(f"Processing resource search query: {query}")
        if image_path:
            logger.info(f"Image path provided: {image_path}")

        # Validate inputs
        if not query and not image_path:
            logger.error("No query or image path provided")
            return {"error": "Either query or image_path must be provided"}

        # Check if image path exists when provided
        if image_path and not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            if not query:
                return {"error": "Image path provided does not exist and no query was provided"}

        # Call search_resources with proper error handling
        result = await search_resources(query, image_path)
        if not result:
            logger.warning("Resource search returned no results")
            return {"error": "Failed to find resources", "message": "Resource search returned no results"}

        # Check if result indicates an error
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            logger.error(f"Resource search returned error: {result[0]['error']}")
            return result[0]

        logger.info(f"Found {len(result)} relevant resources")
        return result

    except Exception as e:
        logger.exception(f"Error processing resource search request: {e}")
        return {
            "error": "Internal server error", 
            "message": str(e)
        }

async def process_module(course_id: Union[int, str], module: Dict, 
                    query: str, course_name: str) -> List[Dict]:
    """Process a single module and return relevant resources.

    This function retrieves all items in a module, analyzes their relevance to the query,
    and returns the most relevant resources with their details.

    Args:
        course_id: The Canvas course ID
        module: The module object containing ID and name
        query: The user's query text
        course_name: The name of the course

    Returns:
        List[Dict]: List of relevant resources with details
    """
    module_id = module["id"]
    module_name = module["name"]

    logger.info(f"Processing module: {module_name} (ID: {module_id})")

    # Get all items in this module
    items = await get_module_items(course_id, module_id)

    if not items:
        logger.warning(f"No items found in module: {module_name}")
        return []

    logger.info(f"Found {len(items)} items in module: {module_name}")

    # Analyze which resources are most relevant within this module
    resource_analysis = analyze_resource_relevance(query, items, course_name, module_name)
    if not resource_analysis:
        logger.warning(f"Failed to analyze resource relevance for module: {module_name}")
        return []

    relevant_indices = resource_analysis.get("resource_indices", [])
    relevance_scores = resource_analysis.get("relevance_scores", [])

    if not relevant_indices:
        logger.warning(f"No relevant resources identified in module: {module_name}")
        return []

    logger.debug(f"Relevant indices from analysis: {relevant_indices}")

    # Add the relevant resources to our results
    relevant_resources = []

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

    # Process file URLs
    for resource in relevant_resources:
        item = resource.pop("item")  # Remove the item from the resource
        if item.get("type") == "File" and "content_id" in item:
            file_url = await get_file_url(course_id, item["content_id"])
            if file_url:
                resource["url"] = file_url

    logger.info(f"Found {len(relevant_resources)} relevant resources in module: {module_name}")
    return relevant_resources

async def search_resources(query: str, image_path: Optional[str] = None) -> List[Dict]:
    """Search for relevant resources across Canvas courses based on a query.

    This function analyzes the query to identify relevant courses and modules,
    then finds the most relevant resources within those modules.

    Args:
        query: The user's query text
        image_path: Optional path to an image file (not supported in current version)

    Returns:
        List[Dict]: List of relevant resources with details, or error information
    """
    # If an image is provided, analyze it to get a query
    if image_path and os.path.exists(image_path):
        logger.info(f"Analyzing image: {image_path}")
        image_query = analyze_image(image_path)
        # Combine the original query with the image analysis if image analysis was successful
        if image_query and query:
            query = f"{query} - {image_query}"
        elif image_query:
            query = image_query
        logger.info(f"Image analysis result: {image_query if image_query else 'Failed to analyze image'}")

    # Step 1: Get all courses
    courses = await get_courses()
    if not courses:
        logger.error("No courses found from Canvas API")
        return [{"error": "No courses found", "details": "Please check your Canvas API configuration"}]

    logger.debug(f"Available courses: {list(courses.keys())}")

    # Step 2: Analyze the query to identify the relevant course
    logger.info(f"Analyzing query: '{query}'")
    course_analysis = analyze_query(query, courses)
    if not course_analysis:
        logger.error("Failed to analyze query to determine relevant course")
        return [{"error": "Failed to analyze query", "details": "Could not determine relevant course"}]

    course_name = course_analysis.get("course_name")
    confidence = course_analysis.get("confidence", 0)
    reasoning = course_analysis.get("reasoning", "No reasoning provided")

    logger.info(f"Course analysis result: {course_name} (confidence: {confidence:.2f})")
    logger.debug(f"Reasoning: {reasoning}")

    # Improved course matching logic if the analyzed course is not found
    if course_name not in courses:
        logger.warning(f"Exact course match not found for '{course_name}', trying partial matches")
        course_name = _find_best_matching_course(query, course_name, courses)

        # Last resort - default to first course
        if course_name not in courses:
            logger.error("No course match found and couldn't determine a relevant course")
            return [{"error": "No relevant course found", "query": query}]

    course_id = courses[course_name]
    logger.info(f"Selected course: {course_name} (ID: {course_id})")

    # Step 3: Get modules for the identified course
    modules = await get_modules(course_id)
    if not modules:
        logger.warning(f"No modules found for course: {course_name}")
        return [{"error": "No modules found", "course": course_name}]

    logger.info(f"Found {len(modules)} modules in course {course_name}")

    # Step 4: Analyze which modules are most relevant
    module_analysis = analyze_query(query, courses, modules)
    if not module_analysis:
        logger.error("Failed to analyze query to determine relevant modules")
        return [{"error": "Failed to analyze query", "details": "Could not determine relevant modules"}]

    relevant_module_names = module_analysis.get("module_names", [])
    logger.info(f"Relevant module names from analysis: {relevant_module_names}")

    # Map module names to modules
    relevant_modules = _find_relevant_modules(modules, relevant_module_names, query)

    # If no relevant modules found
    if not relevant_modules:
        logger.warning("No relevant modules found for the query")
        return [{"error": "No relevant modules found", "course": course_name, "query": query}]

    logger.info(f"Selected {len(relevant_modules)} modules for further analysis")

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
        logger.info(f"Found {len(all_relevant_resources)} relevant resources across all modules")
        return all_relevant_resources
    else:
        logger.warning(f"No relevant resources found for query: {query}")
        return [{"error": "No relevant resources found", "query": query, "course": course_name}]

def _find_best_matching_course(query: str, suggested_course: str, courses: Dict[str, Union[int, str]]) -> str:
    """Find the best matching course when an exact match is not found.

    Args:
        query: The user's query text
        suggested_course: The course name suggested by the analysis
        courses: Dictionary mapping course names to their IDs

    Returns:
        str: The best matching course name
    """
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
            logger.debug(f"Course: {name}, score: {score:.2f}")

            if score > best_match_score:
                best_match_score = score
                best_match = name

        # Also check for simple substring match
        elif suggested_course.lower() in name.lower() or name.lower() in suggested_course.lower():
            logger.debug(f"Substring match found: {name}")
            if best_match is None:
                best_match = name

    if best_match and best_match_score >= 0.2:
        logger.info(f"Using best match course: {best_match} (score: {best_match_score:.2f})")
        return best_match
    elif best_match:
        logger.info(f"Using substring match course: {best_match}")
        return best_match
    else:
        # If still no match found, check if there's a course related to the query topic
        query_keywords = [w.lower() for w in query.split() if len(w) > 4]

        for name in courses:
            for keyword in query_keywords:
                if keyword in name.lower():
                    logger.info(f"Keyword match found: {name} (keyword: {keyword})")
                    return name

        # Last resort - default to first course
        logger.warning("No good match found, defaulting to first course")
        return next(iter(courses.keys()))

def _find_relevant_modules(modules: List[Dict], relevant_module_names: List[str], query: str) -> List[Dict]:
    """Find relevant modules based on module names and query keywords.

    Args:
        modules: List of module objects
        relevant_module_names: List of module names identified as relevant
        query: The user's query text

    Returns:
        List[Dict]: List of relevant module objects
    """
    relevant_modules = []

    # First, find exact matches
    for module in modules:
        if module["name"] in relevant_module_names:
            logger.debug(f"Found exact module match: {module['name']}")
            relevant_modules.append(module)

    # If we can't find exact matches, include modules that contain the query keywords
    if not relevant_modules:
        for module in modules:
            if any(keyword.lower() in module["name"].lower() 
                  for keyword in query.lower().split() if len(keyword) > 3):
                logger.debug(f"Found keyword match in module: {module['name']}")
                relevant_modules.append(module)

    # If still no matches, include the first module as fallback
    if not relevant_modules and modules:
        logger.warning("No matching modules found, using first module as fallback")
        relevant_modules.append(modules[0])

    return relevant_modules

@mcp.tool()
async def get_course_assignments(course_id: Union[int, str], bucket: Optional[str] = None):
    """Use this tool to retrieve all assignments for a specific Canvas course.

    This tool returns assignment details including name, description, due date, and submission status.
    Use this when helping users manage their coursework, check due dates, or find assignment details.

    Args:
        course_id: The Canvas course ID
        bucket: Optional filter - past, overdue, undated, ungraded, unsubmitted, upcoming, future

    Returns:
        List[Dict]: List of assignment objects with details, or None if an error occurs
    """
    # Check cache first
    cache_key = f"{course_id}_{bucket if bucket else 'all'}"
    cached_assignments = Cache.get("assignments", cache_key)
    if cached_assignments:
        logger.info(f"Using cached assignments for course {course_id}")
        return cached_assignments

    try:
        # Get API key
        api_key = Config.get_api_key()
        if not api_key:
            return None

        # Construct API URL
        url = Config.get_api_url(f"courses/{course_id}/assignments")

        # Set up request parameters
        params = {
            "order_by": "due_at",
            "per_page": 100,  # Get max assignments per page
            "include[]": ["submission", "all_dates"]  # Include submission details
        }
        if bucket:
            params["bucket"] = bucket

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Make API request with timeout
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Check response status
        if response.status_code != 200:
            logger.error(f"Canvas API returned status code {response.status_code} for assignments")
            logger.error(f"Response: {response.text}")
            return None

        # Process response
        assignments = response.json()

        # Extract relevant fields
        simplified_assignments = [
            {
                "id": assignment["id"],
                "name": assignment["name"],
                "description": assignment.get("description", ""),
                "due_at": assignment.get("due_at"),
                "has_submitted_submissions": assignment.get("has_submitted_submissions", False)
            } 
            for assignment in assignments
        ]

        # Store in cache
        Cache.set("assignments", simplified_assignments, cache_key)
        logger.info(f"Retrieved {len(simplified_assignments)} assignments for course {course_id}")

        return simplified_assignments

    except requests.RequestException as e:
        logger.error(f"Error connecting to Canvas API for assignments: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_course_assignments: {e}")
        return None

@mcp.tool()
async def get_assignments_by_course_name(course_name: str, bucket: Optional[str] = None):
    """Use this tool to retrieve all assignments for a Canvas course using its name.

    This tool returns assignment details the same as get_course_assignments, but allows
    you to specify the course by name rather than ID. Partial matches are supported.

    Args:
        course_name: The name of the course as it appears in Canvas (partial matches supported)
        bucket: Optional filter - past, overdue, undated, ungraded, unsubmitted, upcoming, future

    Returns:
        List[Dict]: List of assignment objects with details, or None if an error occurs
    """
    # Check cache first
    cache_key = f"{course_name}_{bucket if bucket else 'all'}"
    cached_assignments = Cache.get("assignments", cache_key)
    if cached_assignments:
        logger.info(f"Using cached assignments for course name '{course_name}'")
        return cached_assignments

    try:
        # First get all courses to find the course ID
        courses = await get_courses()
        if not courses:
            logger.error("Could not fetch courses")
            return None

        course_found = False
        course_id = None

        # Try to find a course with a matching name
        for course_name_in_list, course_id_in_list in courses.items():
            if course_name.lower() in course_name_in_list.lower():
                course_id = course_id_in_list
                course_found = True
                logger.info(f"Found matching course: '{course_name_in_list}' (ID: {course_id})")
                break

        # If no course found, return error
        if not course_found:
            logger.warning(f"Course '{course_name}' not found")
            logger.debug(f"Available courses: {list(courses.keys())}")
            return None

        # Get assignments using the course ID
        assignments = await get_course_assignments(course_id, bucket)

        # Store in cache if we got results
        if assignments:
            Cache.set("assignments", assignments, cache_key)

        return assignments

    except Exception as e:
        logger.exception(f"Unexpected error in get_assignments_by_course_name: {e}")
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
    """Search for information across Canvas using natural language queries.

    This tool analyzes the query and returns appropriately formatted results from Canvas.
    Use this for educational queries when you need to find relevant resources, courses,
    or assignments.

    Args:
        query: Natural language query about courses, assignments, or other educational content

    Returns:
        Dict: Formatted search results or error information
    """
    logger.info(f"Searching education platforms with query: '{query}'")

    # Handle general queries about academics or courses
    if "course" in query.lower():
        logger.info("Query contains 'course', fetching course information")
        # Get Canvas courses
        try:
            canvas_courses = await get_courses()
            if canvas_courses:
                logger.info(f"Found {len(canvas_courses)} Canvas courses")
                return {
                    "message": "Here are your Canvas courses:",
                    "results": {"canvas": {"courses": canvas_courses}}
                }
        except Exception as e:
            logger.error(f"Error fetching Canvas courses: {e}")
            return {"error": f"Error fetching Canvas courses: {str(e)}"}

    # Use the Canvas resource finder for other queries
    try:
        logger.info("Using Canvas resource finder for query")
        resources = await find_resources(query=query)

        if isinstance(resources, dict) and "error" in resources:
            logger.warning(f"Resource search returned error: {resources['error']}")
            return resources

        logger.info(f"Found {len(resources) if isinstance(resources, list) else 0} Canvas resources")
        return {
            "message": "Here are the most relevant Canvas resources for your query:",
            "source": "Canvas",
            "resources": resources
        }
    except Exception as e:
        logger.exception(f"Error searching Canvas: {e}")
        return {"error": f"Error searching Canvas: {str(e)}"}

# Test functions
async def test_unified_search():
    """Test the unified search function"""
    logger.info("Testing unified search...")

    # Test a Canvas-specific query
    logger.info("Testing Canvas-specific query...")
    canvas_result = await search_education_platforms("What resources are available for learning matrices in Canvas?")
    logger.info(f"Canvas search result: {canvas_result is not None}")

    # Test a general query
    logger.info("Testing general query...")
    general_result = await search_education_platforms("What courses am I enrolled in?")
    logger.info(f"General search result: {general_result is not None}")

    return {
        "canvas_specific_query_success": canvas_result is not None,
        "general_query_success": general_result is not None
    }

@mcp.tool()
async def get_available_tools():
    """Get a list of all available tools in the Canvas MCP server.

    This tool returns information about all available tools, including their names,
    descriptions, and parameters. Use this to discover what functionality is available.

    Returns:
        Dict: Information about all available tools
    """
    tools = []

    for tool_name, tool in mcp.tools.items():
        tool_info = {
            "name": tool_name,
            "description": tool.description,
            "parameters": {}
        }

        # Extract parameter information if available
        if hasattr(tool, 'parameters'):
            for param_name, param in tool.parameters.items():
                tool_info["parameters"][param_name] = {
                    "type": str(param.type) if hasattr(param, 'type') else "unknown",
                    "required": param.required if hasattr(param, 'required') else False,
                    "description": param.description if hasattr(param, 'description') else ""
                }

        tools.append(tool_info)

    return {
        "count": len(tools),
        "tools": tools
    }

async def run_tests():
    """Run all tests for the Canvas MCP server.

    This function runs tests for various components of the Canvas MCP server,
    including resource search, assignments, and unified search.

    Returns:
        Dict: Test results
    """
    results = {}

    logger.info("=" * 50)
    logger.info("Starting Canvas MCP tests")
    logger.info("=" * 50)

    # Test resource search
    logger.info("Testing resource search...")
    try:
        resources = await find_resources(query="what would be the best resources to learn dot product of matrices from canvas?")
        results["resource_search"] = {
            "success": resources is not None,
            "count": len(resources) if isinstance(resources, list) else 0
        }
        logger.info(f"Resource search test: {'SUCCESS' if resources else 'FAILED'}")
    except Exception as e:
        logger.exception("Resource search test failed")
        results["resource_search"] = {"success": False, "error": str(e)}

    logger.info("=" * 50)

    # Test assignments
    logger.info("Testing assignments...")
    try:
        # Get assignments for a course by name
        assignments = await get_assignments_by_course_name("Linear Algebra")
        results["assignments"] = {
            "success": assignments is not None,
            "count": len(assignments) if assignments else 0
        }
        logger.info(f"Assignments test: {'SUCCESS' if assignments else 'FAILED'}")
    except Exception as e:
        logger.exception("Assignments test failed")
        results["assignments"] = {"success": False, "error": str(e)}

    logger.info("=" * 50)

    # Test unified search
    logger.info("Testing unified search...")
    try:
        unified_search_results = await test_unified_search()
        results["unified_search"] = unified_search_results
        logger.info(f"Unified search test: {'SUCCESS' if unified_search_results['canvas_specific_query_success'] and unified_search_results['general_query_success'] else 'FAILED'}")
    except Exception as e:
        logger.exception("Unified search test failed")
        results["unified_search"] = {"success": False, "error": str(e)}

    logger.info("=" * 50)
    logger.info("All tests completed")

    return results

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Initialize and run the server
        logger.info("Starting Canvas MCP server")
        mcp.run(transport='stdio')
    else:
        # Run tests
        logger.info("Running tests")
        asyncio.run(run_tests())
