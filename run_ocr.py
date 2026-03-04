#!/usr/bin/env python3
"""
Run PaddleOCR on a PDF file to extract text.
"""

import os
import sys

# Try to reduce low-level kernel issues / memory pressure
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

from paddleocr import PaddleOCR
import pypdfium2 as pdfium
import numpy as np
from PIL import Image
import paddle


def pdf_to_images(pdf_path, debug_save_first_page: bool = False):
    """Convert PDF pages to images (numpy arrays). Optionally save first page as PNG for debugging."""
    pdf = pdfium.PdfDocument(pdf_path)
    images = []
    first_page_saved = False

    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            pil_image = page.render(scale=2.0).to_pil()

            if debug_save_first_page and not first_page_saved:
                debug_path = "debug_page1.png"
                pil_image.save(debug_path)
                print(f"Saved first page PNG for debugging to: {debug_path}")
                first_page_saved = True

            # Convert PIL Image to numpy array
            np_image = np.array(pil_image)
            images.append(np_image)
    finally:
        pdf.close()

    return images

def run_ocr_on_pdf(pdf_path, output_file=None):
    """Run OCR on PDF file using PaddleOCR."""
    print("Setting Paddle device to GPU if available...")
    try:
        paddle.set_device("gpu")
        print("Paddle device set to GPU.")
    except Exception as e:
        print(f"Could not use GPU, falling back to CPU: {e}")
        paddle.set_device("cpu")

    print(f"Initializing PaddleOCR...")
    # Initialize PaddleOCR - use_textline_orientation=True for better text detection
    # (this version of PaddleOCR does NOT support a `show_log` argument)
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    except Exception as e:
        print(f"Warning during PaddleOCR init with textline orientation: {e}")
        ocr = PaddleOCR(lang='en')
    
    print(f"Converting PDF to images: {pdf_path}")
    images = pdf_to_images(pdf_path, debug_save_first_page=True)
    print(f"Found {len(images)} pages")
    
    all_results = []
    
    for page_num, image in enumerate(images, 1):
        print(f"\nProcessing page {page_num}/{len(images)}...")
        try:
            result = ocr.predict(image)
        except Exception as e:
            print(f"  Error processing page {page_num}: {e}")
            continue
        
        page_text = []
        # Handle different result formats
        if result:
            # New API format - result is a list of detection results
            for item in result:
                if hasattr(item, 'text') and hasattr(item, 'score'):
                    text = item.text
                    confidence = item.score
                    page_text.append((text, confidence))
                    all_results.append({
                        'page': page_num,
                        'text': text,
                        'confidence': confidence,
                        'bbox': getattr(item, 'bbox', None)
                    })
                elif isinstance(item, dict):
                    # Dictionary format
                    text = item.get('text', '')
                    confidence = item.get('score', item.get('confidence', 0.0))
                    if text:
                        page_text.append((text, confidence))
                        all_results.append({
                            'page': page_num,
                            'text': text,
                            'confidence': confidence,
                            'bbox': item.get('bbox', None)
                        })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Old format: [[bbox], (text, confidence)]
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        page_text.append((text, confidence))
                        all_results.append({
                            'page': page_num,
                            'text': text,
                            'confidence': confidence,
                            'bbox': item[0] if len(item) > 0 else None
                        })
        
        print(f"  Extracted {len(page_text)} text blocks from page {page_num}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"OCR Summary:")
    print(f"  Total pages processed: {len(images)}")
    print(f"  Total text blocks extracted: {len(all_results)}")
    print(f"{'='*60}\n")
    
    # Save results to file if output_file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            current_page = 0
            for result in all_results:
                if result['page'] != current_page:
                    current_page = result['page']
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PAGE {current_page}\n")
                    f.write(f"{'='*60}\n\n")
                f.write(f"{result['text']}\n")
                f.write(f"[Confidence: {result['confidence']:.2f}]\n\n")
        print(f"Results saved to: {output_file}")
    
    # Print first few results
    print("\nFirst few extracted text blocks:")
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i}. [{result['page']}] {result['text'][:80]}... (conf: {result['confidence']:.2f})")
    
    if len(all_results) > 10:
        print(f"... and {len(all_results) - 10} more blocks")
    
    return all_results

if __name__ == "__main__":
    pdf_path = "Enviva_Sample_Tax_Package_10000_units.pdf"
    output_file = "ocr_output.txt"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Running OCR on: {pdf_path}")
    print(f"Output will be saved to: {output_file}\n")
    
    try:
        results = run_ocr_on_pdf(pdf_path, output_file)
        print("\n✓ OCR processing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during OCR processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
