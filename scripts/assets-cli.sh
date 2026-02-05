#!/bin/bash
# Interactive Asset API CLI for ComfyUI
# Usage: ./scripts/assets-cli.sh [BASE_URL]

BASE_URL="${1:-http://127.0.0.1:8188}"
ASSET_ID=""

clear_and_header() {
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║            ComfyUI Asset API Interactive CLI               ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  Base URL: $BASE_URL"
    if [ -n "$ASSET_ID" ]; then
        echo "║  Asset ID: $ASSET_ID"
    fi
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

wait_for_key() {
    echo ""
    echo "Press any key to continue..."
    read -n 1 -s
}

pretty_json() {
    python3 -m json.tool 2>/dev/null || cat
}

show_menu() {
    clear_and_header
    echo "=== SEEDER ==="
    echo "  1) Get seeder status"
    echo "  2) Start seed (async)"
    echo "  3) Start seed (wait for completion)"
    echo "  4) Cancel seed"
    echo ""
    echo "=== ASSETS ==="
    echo "  5) List assets (first 10)"
    echo "  6) List assets (custom query)"
    echo "  7) Get asset detail"
    echo "  8) Download asset content"
    echo "  9) Check if hash exists"
    echo ""
    echo "=== TAGS ==="
    echo " 10) List all tags"
    echo " 11) List tags with prefix"
    echo " 12) Add tags to asset"
    echo " 13) Remove tags from asset"
    echo ""
    echo "=== ASSET MUTATIONS ==="
    echo " 14) Update asset metadata"
    echo " 15) Delete asset"
    echo ""
    echo "=== CONFIG ==="
    echo " 20) Set asset ID"
    echo " 21) Change base URL"
    echo ""
    echo "  q) Quit"
    echo ""
}

# --- Seeder ---

do_seeder_status() {
    echo "=== Seeder Status ==="
    curl -s "$BASE_URL/api/assets/seed/status" | pretty_json
}

do_seed_async() {
    echo "Select roots to scan (comma-separated, e.g., models,input,output):"
    read -r roots_input
    roots_input="${roots_input:-models,input,output}"
    roots_json=$(echo "$roots_input" | tr ',' '\n' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')
    echo "=== Starting Seed (async) ==="
    curl -s -X POST "$BASE_URL/api/assets/seed" \
        -H "Content-Type: application/json" \
        -d "{\"roots\": [$roots_json]}" | pretty_json
}

do_seed_wait() {
    echo "Select roots to scan (comma-separated, e.g., models,input,output):"
    read -r roots_input
    roots_input="${roots_input:-models,input,output}"
    roots_json=$(echo "$roots_input" | tr ',' '\n' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')
    echo "=== Starting Seed (waiting for completion) ==="
    curl -s -X POST "$BASE_URL/api/assets/seed?wait=true" \
        -H "Content-Type: application/json" \
        -d "{\"roots\": [$roots_json]}" | pretty_json
}

do_seed_cancel() {
    echo "=== Cancelling Seed ==="
    curl -s -X POST "$BASE_URL/api/assets/seed/cancel" | pretty_json
}

# --- Assets ---

do_list_assets() {
    echo "=== List Assets (first 10) ==="
    curl -s "$BASE_URL/api/assets?limit=10" | pretty_json
}

do_list_assets_custom() {
    echo "Enter query parameters (e.g., limit=5&sort=size&order=desc&include_tags=models):"
    read -r query
    query="${query:-limit=10}"
    echo "=== List Assets ($query) ==="
    curl -s "$BASE_URL/api/assets?$query" | pretty_json
}

do_get_asset() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "=== Asset Detail: $ASSET_ID ==="
    curl -s "$BASE_URL/api/assets/$ASSET_ID" | pretty_json
}

do_download_asset() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "Enter output path (default: /tmp/asset_download):"
    read -r output_path
    output_path="${output_path:-/tmp/asset_download}"
    echo "=== Downloading Asset: $ASSET_ID ==="
    curl -s -o "$output_path" "$BASE_URL/api/assets/$ASSET_ID/content"
    echo "Downloaded to: $output_path"
    ls -lh "$output_path"
}

do_check_hash() {
    echo "Enter hash (e.g., blake3:abc123...):"
    read -r hash
    echo "=== Checking Hash Exists ==="
    response=$(curl -s -o /dev/null -w "%{http_code}" -I "$BASE_URL/api/assets/hash/$hash")
    if [ "$response" = "200" ]; then
        echo "✓ Hash EXISTS (HTTP 200)"
    elif [ "$response" = "404" ]; then
        echo "✗ Hash NOT FOUND (HTTP 404)"
    else
        echo "? Unexpected response: HTTP $response"
    fi
}

# --- Tags ---

do_list_tags() {
    echo "=== List Tags (first 20) ==="
    curl -s "$BASE_URL/api/tags?limit=20" | pretty_json
}

do_list_tags_prefix() {
    echo "Enter tag prefix (e.g., models):"
    read -r prefix
    echo "=== List Tags with prefix '$prefix' ==="
    curl -s "$BASE_URL/api/tags?prefix=$prefix&limit=20" | pretty_json
}

do_add_tags() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "Enter tags to add (comma-separated, e.g., tag1,tag2):"
    read -r tags_input
    tags_json=$(echo "$tags_input" | tr ',' '\n' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')
    echo "=== Adding Tags to $ASSET_ID ==="
    curl -s -X POST "$BASE_URL/api/assets/$ASSET_ID/tags" \
        -H "Content-Type: application/json" \
        -d "{\"tags\": [$tags_json]}" | pretty_json
}

do_remove_tags() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "Enter tags to remove (comma-separated, e.g., tag1,tag2):"
    read -r tags_input
    tags_json=$(echo "$tags_input" | tr ',' '\n' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')
    echo "=== Removing Tags from $ASSET_ID ==="
    curl -s -X DELETE "$BASE_URL/api/assets/$ASSET_ID/tags" \
        -H "Content-Type: application/json" \
        -d "{\"tags\": [$tags_json]}" | pretty_json
}

# --- Asset Mutations ---

do_update_asset() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "Enter new name (leave empty to skip):"
    read -r new_name
    echo "Enter user_metadata as JSON (e.g., {\"key\": \"value\"}, leave empty to skip):"
    read -r metadata
    
    payload="{"
    if [ -n "$new_name" ]; then
        payload="$payload\"name\": \"$new_name\""
    fi
    if [ -n "$metadata" ]; then
        if [ -n "$new_name" ]; then
            payload="$payload, "
        fi
        payload="$payload\"user_metadata\": $metadata"
    fi
    payload="$payload}"
    
    echo "=== Updating Asset $ASSET_ID ==="
    echo "Payload: $payload"
    curl -s -X PATCH "$BASE_URL/api/assets/$ASSET_ID" \
        -H "Content-Type: application/json" \
        -d "$payload" | pretty_json
}

do_delete_asset() {
    if [ -z "$ASSET_ID" ]; then
        echo "Enter asset ID (UUID):"
        read -r input_id
        ASSET_ID="$input_id"
    fi
    echo "Delete file content too? (y/n, default: n):"
    read -r delete_content
    delete_param="false"
    if [ "$delete_content" = "y" ] || [ "$delete_content" = "Y" ]; then
        delete_param="true"
    fi
    echo "=== Deleting Asset $ASSET_ID (delete_content=$delete_param) ==="
    response=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/api/assets/$ASSET_ID?delete_content=$delete_param")
    if [ "$response" = "204" ]; then
        echo "✓ Asset deleted successfully (HTTP 204)"
        ASSET_ID=""
    elif [ "$response" = "404" ]; then
        echo "✗ Asset not found (HTTP 404)"
    else
        echo "? Unexpected response: HTTP $response"
    fi
}

# --- Config ---

do_set_asset_id() {
    echo "Current asset ID: ${ASSET_ID:-<not set>}"
    echo "Enter new asset ID (or 'clear' to unset):"
    read -r input_id
    if [ "$input_id" = "clear" ]; then
        ASSET_ID=""
        echo "Asset ID cleared."
    else
        ASSET_ID="$input_id"
        echo "Asset ID set to: $ASSET_ID"
    fi
}

do_set_base_url() {
    echo "Current base URL: $BASE_URL"
    echo "Enter new base URL:"
    read -r new_url
    if [ -n "$new_url" ]; then
        BASE_URL="$new_url"
        echo "Base URL set to: $BASE_URL"
    fi
}

# --- Main Loop ---

while true; do
    show_menu
    echo -n "Select option: "
    read -r choice
    
    clear_and_header
    
    case $choice in
        1)  do_seeder_status ;;
        2)  do_seed_async ;;
        3)  do_seed_wait ;;
        4)  do_seed_cancel ;;
        5)  do_list_assets ;;
        6)  do_list_assets_custom ;;
        7)  do_get_asset ;;
        8)  do_download_asset ;;
        9)  do_check_hash ;;
        10) do_list_tags ;;
        11) do_list_tags_prefix ;;
        12) do_add_tags ;;
        13) do_remove_tags ;;
        14) do_update_asset ;;
        15) do_delete_asset ;;
        20) do_set_asset_id ;;
        21) do_set_base_url ;;
        q|Q) 
            echo "Goodbye!"
            exit 0 
            ;;
        *)  
            echo "Invalid option: $choice"
            ;;
    esac
    
    wait_for_key
done
