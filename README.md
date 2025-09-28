# QR Code File Backup Scripts

Python-based scripts for encoding binary files into QR codes stored in PDF format, and decoding them back to the original files. This creates a physical, printable backup of your files that can be reconstructed from scanned copies.

**Note**: This system is designed for relatively small files and should be tested thoroughly before relying on it for critical data backup. Always maintain multiple backup strategies for important files. Make sure that the generated PDF can be properly reconstructed from a scan.

## Overview

This system consists of two complementary tools:

1. **`binary_to_qr.py`** - Encoder that converts any binary file into a PDF filled with QR codes
2. **`qr_pdf_decoder.py`** - Decoder that reconstructs the original file from the QR code PDF

The system is designed to be robust against physical degradation - you can print the PDF, scan it later (even with a phone camera), and still recover your original file with built-in integrity verification. Note that all QR codes must
be present for decoding. The encoding does not support losing any of them.

## Features

### Encoder Features
- **Flexible chunk sizes**: Adjustable data per QR code
- **High error correction**: Uses QR code error correction (30% recovery)
- **File integrity**: SHA-256 hash verification for data integrity

### Decoder Features
- **Robust scanning**: Handles scanned, photographed, or low-quality PDFs
- **Multiple preprocessing**: Various image enhancement techniques for better QR detection
- **Integrity verification**: Automatic SHA-256 hash validation

## Installation

Python 3.12 or higher is required.

```bash
sudo apt-get install poppler-utils libzbar-dev
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Encoding Files to QR Codes

```bash
python binary_to_qr.py input_file.dat output_backup.pdf
```

### Decoding QR Codes Back to Files

```bash
python qr_pdf_decoder.py backup.pdf restored_file.dat
```

## How It Works

### Encoding Process

1. **File Reading**: The input file is read as binary data
2. **Integrity Hash**: SHA-256 hash is calculated for verification
3. **Data Chunking**: File is split into chunks (default 512 bytes each)
4. **QR Generation**: Each chunk is:
   - Base64 encoded for QR compatibility
   - Combined with metadata (filename, hash, chunk info)
   - Encoded into a high-error-correction QR code
5. **PDF Creation**: QR codes are arranged in a clean PDF layout with:
   - File metadata headers
   - Page numbering
   - Individual QR code labels

### Decoding Process

1. **PDF Processing**: PDF pages are converted to high-resolution images
2. **QR Detection**: Multiple image preprocessing approaches detect QR codes:
   - Direct detection on original image
   - Adaptive threshold preprocessing
   - Binary threshold with OTSU method
3. **Data Parsing**: Each QR code is decoded and parsed for:
   - File metadata validation
   - Chunk ordering information
   - Base64 data decoding
4. **File Reconstruction**: Chunks are assembled in correct order
5. **Integrity Verification**: SHA-256 hash confirms perfect reconstruction

### Data Format

Each QR code contains data in this format:
```
filename|sha256_hash|chunk_index|total_chunks|chunk_size|base64_data
```

Example:
```
document.pdf|a1b2c3d4...|0|15|2000|iVBORw0KGgoAAAANSUhEUgAA...
```

### Troubleshooting

**"No QR codes found":**
- Increase scan resolution (try 300+ DPI)
- Improve lighting when scanning
- Ensure QR codes are not cropped or distorted
- Use `--verbose` flag to see detection details

**"Hash verification failed":**
- One or more QR codes were decoded incorrectly
- Try rescanning with different preprocessing
- Check for systematic scanning issues (blur, skew)

**"Missing chunks":**
- Some QR codes couldn't be detected or decoded
- The verbose output will show which chunks are missing
- Focus on rescanning specific pages with missing codes

## Best Practices

### For Encoding
- **Test first**: Try with a small test file to verify the process
- **Choose appropriate chunk size**: Smaller chunks = more resilient but more QR codes
- **Document the process**: Keep notes about encoding parameters used

### For Physical Backup
- **High-quality printing**: Use good printers with sufficient contrast
- **Archival paper**: Consider acid-free paper for long-term storage
- **Multiple copies**: Print several copies for redundancy
- **Protective storage**: Keep away from light, moisture, and physical damage

### For Scanning/Recovery
- **High resolution**: Scan at 300 DPI or higher
- **Good lighting**: Even, bright lighting without glare
- **Steady scanning**: Avoid blur from camera shake
- **Clean scanner**: Remove dust and smudges from scanner glass

## Limitations

- **File size**: Best suited for files under 100KB due to PDF size growth
- **Print quality**: Requires good quality printing and scanning
- **Processing time**: Large files take significant time to encode/decode
- **Storage efficiency**: Very inefficient compared to digital storage

## Security Considerations

- **No encryption**: Files are encoded as-is without encryption
- **Visible metadata**: Filenames and hashes are visible in the PDF
- **Physical security**: Printed copies are physically accessible
- **Add encryption**: Consider encrypting files before encoding for sensitive data

## Examples and Demos

### Quick Test
```bash
# Create a test file
echo "Hello, QR Code backup!" > test.txt

# Encode it
python binary_to_qr.py test.txt test_backup.pdf

# Decode it back
python qr_pdf_decoder.py test_backup.pdf test_restored.txt

# Verify
diff test.txt test_restored.txt  # Should show no differences
```
