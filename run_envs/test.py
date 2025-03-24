import json

tools = [{
    "name": "search_shop",
    "description": "Search for a range of items in shop database",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "Item to search for"
            },
            "range": {
                "type": "string",
                "description": "Range of results to return"
            }
        },
        "required": ["keywords", "range"]
    },
}]

print(repr(json.dumps(tools)))