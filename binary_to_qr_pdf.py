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
Binary File to QR Code PDF Encoder

This script takes a binary file as input and encodes it as a sequence of QR codes,
which are then written to a PDF file. Each QR code contains a chunk of the binary data
along with metadata for reconstruction.
"""

import argparse
import base64
import hashlib
import math
import os
import sys
from io import BytesIO
from typing import List, Tuple

import qrcode
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class BinaryToQREncoder:
    def __init__(self, chunk_size: int = 2000):
        """
        Initialize the encoder.

        Args:
            chunk_size: Maximum bytes per QR code (default: 2000)
                       QR codes can handle up to ~2953 bytes in binary mode,
                       but we use base64 encoding which increases size by ~33%
        """
        self.chunk_size = chunk_size

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the file for integrity verification."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def split_binary_data(self, data: bytes) -> List[bytes]:
        """Split binary data into chunks suitable for QR codes."""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunks.append(data[i:i + self.chunk_size])
        return chunks

    def create_qr_data(self, chunk: bytes, chunk_index: int, total_chunks: int,
                      file_hash: str, original_filename: str) -> str:
        """
        Create QR code data with metadata and base64-encoded chunk.

        Format: HEADER|BASE64_DATA
        Header format: filename|hash|chunk_index|total_chunks|chunk_size
        """
        # Encode chunk as base64 to ensure QR code compatibility
        encoded_chunk = base64.b64encode(chunk).decode('ascii')

        # Create header with metadata
        header = f"{original_filename}|{file_hash}|{chunk_index}|{total_chunks}|{len(chunk)}"

        # Combine header and data
        qr_data = f"{header}|{encoded_chunk}"

        return qr_data

    def create_qr_code(self, data: str) -> Image.Image:
        """Create a QR code image from data string."""
        qr = qrcode.QRCode(
            version=None,  # Auto-determine version based on data
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        return qr_image

    def create_pdf(self, qr_images: List[Image.Image], output_path: str,
                  original_filename: str, file_hash: str):
        """Create PDF with QR codes arranged on pages."""
        c = canvas.Canvas(output_path, pagesize=A4)
        page_width, page_height = A4

        # QR code dimensions and layout
        qr_size = 200  # Size of each QR code in points
        margin = 50
        cols = int((page_width - 2 * margin) // qr_size)
        rows = int((page_height - 2 * margin - 100) // qr_size)  # Leave space for header
        qr_per_page = cols * rows

        total_qr_codes = len(qr_images)
        total_pages = math.ceil(total_qr_codes / qr_per_page)

        for page_num in range(total_pages):
            # Add page header
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, page_height - 30, f"File: {original_filename}")
            c.setFont("Helvetica", 10)
            c.drawString(margin, page_height - 50, f"SHA-256: {file_hash}")
            c.drawString(margin, page_height - 70,
                        f"Page {page_num + 1} of {total_pages} | "
                        f"Total QR Codes: {total_qr_codes}")

            # Add QR codes to page
            start_idx = page_num * qr_per_page
            end_idx = min(start_idx + qr_per_page, total_qr_codes)

            for i, qr_idx in enumerate(range(start_idx, end_idx)):
                qr_image = qr_images[qr_idx]

                # Calculate position
                col = i % cols
                row = i // cols
                x = margin + col * qr_size
                y = page_height - margin - 100 - (row + 1) * qr_size

                # Convert PIL image to ReportLab ImageReader
                img_buffer = BytesIO()
                qr_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_reader = ImageReader(img_buffer)

                # Draw QR code
                c.drawImage(img_reader, x, y, width=qr_size, height=qr_size)

                # Add QR code index label
                c.setFont("Helvetica", 8)
                c.drawString(x, y - 15, f"QR {qr_idx + 1}/{total_qr_codes}")

            c.showPage()

        c.save()

    def encode_file(self, input_file: str, output_pdf: str) -> Tuple[int, str]:
        """
        Main method to encode a binary file to QR codes in PDF.

        Returns:
            Tuple of (number of QR codes created, file hash)
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read binary file
        with open(input_file, 'rb') as f:
            binary_data = f.read()

        # Calculate file hash for integrity
        file_hash = self.calculate_file_hash(input_file)
        original_filename = os.path.basename(input_file)

        print(f"Encoding file: {original_filename}")
        print(f"File size: {len(binary_data):,} bytes")
        print(f"SHA-256: {file_hash}")

        # Split data into chunks
        chunks = self.split_binary_data(binary_data)
        total_chunks = len(chunks)

        print(f"Creating {total_chunks} QR codes with {self.chunk_size} bytes per code...")

        # Create QR codes
        qr_images = []
        for i, chunk in enumerate(chunks):
            qr_data = self.create_qr_data(chunk, i, total_chunks, file_hash, original_filename)
            qr_image = self.create_qr_code(qr_data)
            qr_images.append(qr_image)
            print(f"Created QR code {i + 1}/{total_chunks}")

        # Create PDF
        print(f"Generating PDF: {output_pdf}")
        self.create_pdf(qr_images, output_pdf, original_filename, file_hash)

        print(f"Successfully created PDF with {total_chunks} QR codes")
        return total_chunks, file_hash


def main():
    parser = argparse.ArgumentParser(
        description="Encode a binary file as QR codes in a PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python binary_to_qr.py input.bin output.pdf
  python binary_to_qr.py --chunk-size 512 document.pdf backup.pdf
  python binary_to_qr.py image.jpg image_qr_backup.pdf
        """
    )

    parser.add_argument("input_file", help="Path to the binary file to encode")
    parser.add_argument("output_pdf", help="Path for the output PDF file")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Bytes per QR code (default: 512)")

    args = parser.parse_args()

    try:
        encoder = BinaryToQREncoder(chunk_size=args.chunk_size)
        qr_count, file_hash = encoder.encode_file(args.input_file, args.output_pdf)

        # Print summary
        file_size = os.path.getsize(args.input_file)
        pdf_size = os.path.getsize(args.output_pdf)

        print("\n" + "="*50)
        print("ENCODING COMPLETE")
        print("="*50)
        print(f"Input file: {args.input_file}")
        print(f"Output PDF: {args.output_pdf}")
        print(f"Original size: {file_size:,} bytes")
        print(f"PDF size: {pdf_size:,} bytes")
        print(f"QR codes created: {qr_count}")
        print(f"Expansion ratio: {pdf_size/file_size:.1f}x")
        print(f"File hash: {file_hash}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()