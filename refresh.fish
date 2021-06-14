#!/usr/bin/env fish
# removes and re-adds binary content to remove from git cache

set FILES **/*.{pdf,png,jpg}
# not sure how to turn a glob into a string
set GLOB "*.{pdf,png,jpg}"
set TMP "tmp"
set LEN (math "1 + "(string length TMP))

# rename files to temp name
for path in $FILES
  mv "$path" "$path.$TMP"
end

# run git commands
# git ci "remove binary files"
# git add --all
# git push
# bfg --delete-files "$GLOB"

# revert file names
# for path in **/*.$TMP
#   mv "$path" (string sub --end="-$LEN" -- "$path")
# end

