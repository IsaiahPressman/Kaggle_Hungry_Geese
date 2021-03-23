archive_name="${1:-"submission.tar.gz"}"
tar --exclude="*__pycache__*" -czvf "$archive_name" main.py cp.pt hungry_geese
echo "Agent saved to $archive_name"