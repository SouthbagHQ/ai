import os

filepath = 'southbag-online-bot/src/prompt.js'
out_file = 'input.txt'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract just the prompt text between the backticks
start = content.find('`') + 1
end = content.rfind('`')

if start > 0 and end > start:
    text = content[start:end]
else:
    text = content

with open(out_file, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Extracted system prompt! Wrote {len(text)} characters to {out_file}")
