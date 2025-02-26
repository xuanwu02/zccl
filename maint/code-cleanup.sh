#!/bin/bash
set -euo pipefail

# Format all C/C++ files in the project
format_all() {
    echo "Formatting all C/C++ files..."
    find ZCCL/ hZ-dynamic/src/ -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} \;
}

# Format only files tracked by git
format_git() {
    echo "Formatting git-tracked C/C++ files..."
    git ls-files '*.c' '*.h' | xargs clang-format -i
}

# Format only staged files
format_staged() {
    echo "Formatting staged C/C++ files..."
    STAGED_FILES=$(git diff --cached --name-only --diff-filter=d -- '*.c' '*.h')
    if [ -z "$STAGED_FILES" ]; then
        echo "No staged C/C++ files to format."
        return
    fi
    echo "$STAGED_FILES" | xargs -r clang-format -i
    echo "$STAGED_FILES" | xargs -r git add
    echo "Formatted and re-staged changes"
}

# Dry run to check formatting
dry_run() {
    echo "Checking formatting..."
    find ZCCL/ hZ-dynamic/src/ -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format --Werror --dry-run {} \;
}

# Show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Code formatting utility script"
    echo ""
    echo "Options:"
    echo "  --all      Format all C/C++ files (default)"
    echo "  --git      Format only git-tracked files" 
    echo "  --staged   Format only staged files and re-stage changes"
    echo "  --dry-run  Check formatting without changes"
    echo "  --help     Show this help message"
}

# Main script execution
case "${1:-}" in  # Use parameter expansion to handle unset variable
    ""|--all)
        format_all
        ;;
    --git)
        format_git
        ;;
    --staged)
        format_staged
        ;;
    --dry-run)
        dry_run
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Error: Unknown option $1"
        show_help
        exit 1
        ;;
esac

echo "Formatting completed successfully"
