
# 🔄 Atom of Thoughts: Auto-Sync Utility

# Add all changes
git add .

# Get current timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Commit with timestamp
git commit -m "Auto-update: $timestamp"

# Push to origin
git push origin main

Write-Host "✅ Project synced to GitHub at $timestamp" -ForegroundColor Green
