#!/bin/bash
# Asset API testing script for ComfyUI
# Usage: ./scripts/test-assets-api.sh [BASE_URL]

BASE_URL="${1:-http://127.0.0.1:8188}"

echo "=== ComfyUI Asset API Test Script ==="
echo "Base URL: $BASE_URL"
echo ""

# --- Seeder Status & Control ---

echo "--- Seeder Status ---"
curl -s "$BASE_URL/api/assets/seed/status" | python3 -m json.tool
echo ""

echo "--- Start Seed (async) ---"
curl -s -X POST "$BASE_URL/api/assets/seed" \
  -H "Content-Type: application/json" \
  -d '{"roots": ["models", "input", "output"]}' | python3 -m json.tool
echo ""

echo "--- Start Seed (wait for completion) ---"
curl -s -X POST "$BASE_URL/api/assets/seed?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"roots": ["models"]}' | python3 -m json.tool
echo ""

echo "--- Cancel Seed ---"
curl -s -X POST "$BASE_URL/api/assets/seed/cancel" | python3 -m json.tool
echo ""

# --- List Assets ---

echo "--- List Assets (first 10) ---"
curl -s "$BASE_URL/api/assets?limit=10" | python3 -m json.tool
echo ""

echo "--- List Assets with tag filter ---"
curl -s "$BASE_URL/api/assets?include_tags=models&limit=5" | python3 -m json.tool
echo ""

echo "--- List Assets sorted by size ---"
curl -s "$BASE_URL/api/assets?sort=size&order=desc&limit=5" | python3 -m json.tool
echo ""

# --- Tags ---

echo "--- List Tags ---"
curl -s "$BASE_URL/api/tags?limit=20" | python3 -m json.tool
echo ""

echo "--- List Tags with prefix ---"
curl -s "$BASE_URL/api/tags?prefix=models&limit=10" | python3 -m json.tool
echo ""

# --- Single Asset Operations (requires valid asset ID) ---
# Uncomment and replace ASSET_ID with a real UUID from list assets

# ASSET_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
# 
# echo "--- Get Asset Detail ---"
# curl -s "$BASE_URL/api/assets/$ASSET_ID" | python3 -m json.tool
# echo ""
# 
# echo "--- Check Asset Exists by Hash ---"
# curl -s -I "$BASE_URL/api/assets/hash/blake3:abc123..."
# echo ""
# 
# echo "--- Download Asset Content ---"
# curl -s -o /tmp/downloaded_asset "$BASE_URL/api/assets/$ASSET_ID/content"
# echo "Downloaded to /tmp/downloaded_asset"
# echo ""
# 
# echo "--- Add Tags to Asset ---"
# curl -s -X POST "$BASE_URL/api/assets/$ASSET_ID/tags" \
#   -H "Content-Type: application/json" \
#   -d '{"tags": ["my-tag", "another-tag"]}' | python3 -m json.tool
# echo ""
# 
# echo "--- Remove Tags from Asset ---"
# curl -s -X DELETE "$BASE_URL/api/assets/$ASSET_ID/tags" \
#   -H "Content-Type: application/json" \
#   -d '{"tags": ["my-tag"]}' | python3 -m json.tool
# echo ""
# 
# echo "--- Update Asset Metadata ---"
# curl -s -X PATCH "$BASE_URL/api/assets/$ASSET_ID" \
#   -H "Content-Type: application/json" \
#   -d '{"name": "New Name", "user_metadata": {"key": "value"}}' | python3 -m json.tool
# echo ""
# 
# echo "--- Delete Asset ---"
# curl -s -X DELETE "$BASE_URL/api/assets/$ASSET_ID?delete_content=false"
# echo ""

echo "=== Done ==="
