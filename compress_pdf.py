import os
import re
import sys
import fitz  # PyMuPDF

def clean_text(text: str) -> str:
    """
    Apply structural cleaning to the text:
    1. Consolidate multiple spaces and line breaks.
    2. Strip out lines that look like page numbers (e.g. "Page 1 of 10" or "1").
    3. Remove double-spacing.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Split into lines to evaluate structural elements
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Skip page numbers (e.g., "1", "Page 1", "1 / 10", "Page 1 of 5")
        if re.match(r'^(page\s+)?\d+(\s*(of|/)\s*\d+)?$', stripped, re.IGNORECASE):
            continue
            
        cleaned_lines.append(stripped)
        
    return "\n".join(cleaned_lines)

def compress_pdf(input_path: str, output_path: str):
    """
    Read an input PDF, extract and clean the text of each page, 
    and write the cleaned text into a new, smaller PDF.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Reading document: {input_path}")
    doc = fitz.open(input_path)
    out_doc = fitz.open()  # Create a brand new empty PDF
    
    total_chars_original = 0
    total_chars_cleaned = 0
    
    for i, page in enumerate(doc):
        text = page.get_text()
        total_chars_original += len(text)
        
        # Clean the page text
        cleaned = clean_text(text)
        total_chars_cleaned += len(cleaned)
        
        # Create a new corresponding page in the output PDF
        # We use a standard A4 page size
        new_page = out_doc.new_page(width=595, height=842)
        
        # Insert the cleaned text onto the page
        # margin of 50 points from top-left
        new_page.insert_htmlbox(fitz.Rect(50, 50, 545, 792), f"<div style='font-family:sans-serif; font-size:10px; line-height:1.4;'>{cleaned.replace(chr(10), '<br>')}</div>")
        
    out_doc.save(output_path)
    out_doc.close()
    doc.close()
    
    savings = ((total_chars_original - total_chars_cleaned) / total_chars_original) * 100 if total_chars_original > 0 else 0
    print(f"✓ Compression complete!")
    print(f"Original characters: {total_chars_original}")
    print(f"Compressed characters: {total_chars_cleaned}")
    print(f"Character reduction: {savings:.1f}%")
    print(f"Saved compressed file to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compress_pdf.py [input_file.pdf] [output_file.pdf]")
        sys.exit(1)
        
    compress_pdf(sys.argv[1], sys.argv[2])
