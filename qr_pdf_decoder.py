# MIT License
#
# Copyright (c) 2025 Vitaly Chipounov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/env python3
"""
QR Code PDF to Binary File Decoder

This script takes a PDF file containing QR codes (created by the binary_to_qr.py encoder)
and decodes it back to the original binary file. It can handle scanned PDFs and images.
"""

import argparse
import base64
import hashlib
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from pyzbar import pyzbar
from pdf2image import convert_from_path


class QRChunk:
    """Represents a decoded QR code chunk with metadata."""

    def __init__(self, filename: str, file_hash: str, chunk_index: int,
                 total_chunks: int, chunk_size: int, data: bytes):
        self.filename = filename
        self.file_hash = file_hash
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.chunk_size = chunk_size
        self.data = data


class QRPDFDecoder:
    def __init__(self, verbose: bool = False):
        """
        Initialize the decoder.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Returns:
            List of images as numpy arrays
        """
        self.log(f"Converting PDF to images: {pdf_path}")

        try:
            # Convert PDF to PIL images
            pages = convert_from_path(pdf_path, dpi=300, fmt='RGB')

            # Convert PIL images to OpenCV format (numpy arrays)
            images = []
            for i, page in enumerate(pages):
                # Convert PIL to numpy array (RGB)
                img_array = np.array(page)
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
                self.log(f"Converted page {i+1}/{len(pages)}")

            return images

        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better QR code detection.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding to handle varying lighting
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return processed

    def detect_qr_codes(self, image: np.ndarray) -> List[str]:
        """
        Detect and decode QR codes in an image.

        Args:
            image: Input image as numpy array

        Returns:
            List of decoded QR code data strings
        """
        qr_data = []

        # Try original image first
        barcodes = pyzbar.decode(image)

        # If no QR codes found, try preprocessed image
        if not barcodes:
            processed_image = self.preprocess_image(image)
            barcodes = pyzbar.decode(processed_image)

        # If still no codes, try different preprocessing
        if not barcodes:
            # Try different threshold values
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Binary threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            barcodes = pyzbar.decode(thresh)

        for barcode in barcodes:
            # Decode barcode data
            barcode_data = barcode.data.decode('utf-8')
            qr_data.append(barcode_data)

            if self.verbose:
                # Get barcode location for debugging
                (x, y, w, h) = barcode.rect
                self.log(f"Found QR code at ({x}, {y}) size {w}x{h}")

        return qr_data

    def parse_qr_data(self, qr_data: str) -> Optional[QRChunk]:
        """
        Parse QR code data string into QRChunk object.

        Expected format: filename|hash|chunk_index|total_chunks|chunk_size|base64_data

        Args:
            qr_data: Raw QR code data string

        Returns:
            QRChunk object or None if parsing fails
        """
        try:
            parts = qr_data.split('|')
            if len(parts) != 6:
                self.log(f"Invalid QR data format: expected 6 parts, got {len(parts)}")
                return None

            filename = parts[0]
            file_hash = parts[1]
            chunk_index = int(parts[2])
            total_chunks = int(parts[3])
            chunk_size = int(parts[4])
            base64_data = parts[5]

            # Decode base64 data
            try:
                chunk_data = base64.b64decode(base64_data)
            except Exception as e:
                self.log(f"Failed to decode base64 data: {e}")
                return None

            return QRChunk(filename, file_hash, chunk_index, total_chunks, chunk_size, chunk_data)

        except Exception as e:
            self.log(f"Error parsing QR data: {e}")
            return None

    def decode_pdf(self, pdf_path: str) -> Dict[int, QRChunk]:
        """
        Decode all QR codes from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping chunk index to QRChunk objects
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        chunks = {}
        total_found = 0

        for page_num, image in enumerate(images):
            self.log(f"Processing page {page_num + 1}/{len(images)}")

            # Detect QR codes in image
            qr_data_list = self.detect_qr_codes(image)

            self.log(f"Found {len(qr_data_list)} QR codes on page {page_num + 1}")

            # Parse each QR code
            for qr_data in qr_data_list:
                chunk = self.parse_qr_data(qr_data)
                if chunk:
                    if chunk.chunk_index in chunks:
                        self.log(f"Warning: Duplicate chunk {chunk.chunk_index} found")
                    else:
                        chunks[chunk.chunk_index] = chunk
                        total_found += 1
                        self.log(f"Successfully decoded chunk {chunk.chunk_index}")

        print(f"Decoded {total_found} QR codes from {len(images)} pages")

        return chunks

    def reconstruct_file(self, chunks: Dict[int, QRChunk], output_path: str) -> bool:
        """
        Reconstruct the original file from decoded chunks.

        Args:
            chunks: Dictionary of chunk index to QRChunk objects
            output_path: Path where to save the reconstructed file

        Returns:
            True if reconstruction successful, False otherwise
        """
        if not chunks:
            print("Error: No chunks found to reconstruct")
            return False

        # Get file metadata from first chunk
        first_chunk = next(iter(chunks.values()))
        filename = first_chunk.filename
        expected_hash = first_chunk.file_hash
        total_chunks = first_chunk.total_chunks

        print(f"Reconstructing file: {filename}")
        print(f"Expected chunks: {total_chunks}")
        print(f"Found chunks: {len(chunks)}")
        print(f"Expected SHA-256: {expected_hash}")

        # Check if we have all chunks
        missing_chunks = []
        for i in range(total_chunks):
            if i not in chunks:
                missing_chunks.append(i)

        if missing_chunks:
            print(f"Error: Missing chunks: {missing_chunks}")
            return False

        # Verify all chunks have consistent metadata
        for chunk in chunks.values():
            if (chunk.filename != filename or
                chunk.file_hash != expected_hash or
                chunk.total_chunks != total_chunks):
                print("Error: Inconsistent chunk metadata")
                return False

        # Reconstruct file data
        print("Reconstructing file data...")
        file_data = bytearray()

        for i in range(total_chunks):
            chunk = chunks[i]
            file_data.extend(chunk.data)
            print(f"Processed chunk {i + 1}/{total_chunks}")

        # Verify file integrity
        print("Verifying file integrity...")
        calculated_hash = hashlib.sha256(file_data).hexdigest()

        if calculated_hash != expected_hash:
            print(f"Error: File hash mismatch!")
            print(f"Expected: {expected_hash}")
            print(f"Calculated: {calculated_hash}")
            return False

        # Write reconstructed file
        print(f"Writing reconstructed file: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(file_data)

        print(f"Successfully reconstructed file: {len(file_data):,} bytes")
        print(f"SHA-256 verified: {calculated_hash}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Decode QR codes from PDF back to original binary file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qr_pdf_decoder.py backup.pdf restored_file.bin
  python qr_pdf_decoder.py --verbose document_qr_backup.pdf document_restored.pdf
  python qr_pdf_decoder.py scanned_backup.pdf image_restored.jpg
        """
    )

    parser.add_argument("input_pdf", help="Path to the PDF file with QR codes")
    parser.add_argument("output_file", help="Path for the reconstructed file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        decoder = QRPDFDecoder(verbose=args.verbose)

        # Check if input file exists
        if not os.path.exists(args.input_pdf):
            raise FileNotFoundError(f"Input PDF not found: {args.input_pdf}")

        print(f"Decoding QR codes from: {args.input_pdf}")

        # Decode PDF
        chunks = decoder.decode_pdf(args.input_pdf)

        if not chunks:
            print("Error: No valid QR codes found in PDF")
            sys.exit(1)

        # Reconstruct file
        success = decoder.reconstruct_file(chunks, args.output_file)

        if success:
            # Print summary
            input_size = os.path.getsize(args.input_pdf)
            output_size = os.path.getsize(args.output_file)

            print("\n" + "="*50)
            print("DECODING COMPLETE")
            print("="*50)
            print(f"Input PDF: {args.input_pdf}")
            print(f"Output file: {args.output_file}")
            print(f"PDF size: {input_size:,} bytes")
            print(f"Reconstructed size: {output_size:,} bytes")
            print(f"Chunks decoded: {len(chunks)}")
        else:
            print("Failed to reconstruct file")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
