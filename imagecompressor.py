"""
Advanced Image Compressor with Multiple Compression Techniques
===========================================================

This script implements a sophisticated image compression system that combines
multiple techniques to achieve optimal file size reduction while maintaining
acceptable visual quality. It's like having a Swiss Army knife for image compression!

Author: Your Friendly Neighborhood Coder
Date: Because pixels deserve better treatment
"""

import os
import sys
import argparse
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import time
import logging
from typing import Tuple, List, Optional, Dict
import json

# Set up logging because we want to know what's happening under the hood
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compression.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageCompressor:
    """
    A comprehensive image compression class that's smarter than your average bear.
    
    This class handles multiple compression strategies:
    1. Quality-based JPEG compression (the classic approach)
    2. Intelligent resizing (because sometimes less is more)
    3. Color palette optimization (for those who appreciate efficiency)
    4. Progressive JPEG encoding (for web-friendly loading)
    5. Format conversion optimization (choosing the right tool for the job)
    """
    
    def __init__(self, target_size_kb: Optional[int] = None, 
                 max_dimension: Optional[int] = None,
                 preserve_aspect_ratio: bool = True):
        """
        Initialize our compression wizard with sensible defaults.
        
        Args:
            target_size_kb: Target file size in KB. If None, we'll use quality-based compression
            max_dimension: Maximum width or height in pixels. Larger images will be resized
            preserve_aspect_ratio: Whether to maintain the original image proportions
        """
        self.target_size_kb = target_size_kb
        self.max_dimension = max_dimension
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # These are our compression quality presets - think of them as difficulty levels
        self.quality_presets = {
            'maximum': 95,    # For when quality is everything
            'high': 85,       # Sweet spot for most use cases
            'medium': 70,     # Balanced approach
            'low': 50,        # Aggressive compression
            'minimum': 30     # When file size matters more than your eyeballs
        }
        
        # Supported formats - our toolbox of image types
        self.supported_formats = {
            'JPEG': ['.jpg', '.jpeg'],
            'PNG': ['.png'],
            'WEBP': ['.webp'],
            'BMP': ['.bmp'],
            'TIFF': ['.tiff', '.tif']
        }
        
        logger.info(f"ImageCompressor initialized with target_size: {target_size_kb}KB, "
                   f"max_dimension: {max_dimension}px")
    
    def _get_file_size_kb(self, file_path: str) -> float:
        """
        Get file size in KB because humans think in KB, not bytes.
        
        Args:
            file_path: Path to the file we're checking
            
        Returns:
            File size in kilobytes (rounded to 2 decimal places)
        """
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        return round(size_kb, 2)
    
    def _calculate_new_dimensions(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate new image dimensions while keeping things proportional.
        It's like resizing a photo frame - we want to keep the picture looking right!
        
        Args:
            original_size: Tuple of (width, height) in pixels
            
        Returns:
            New dimensions as (width, height) tuple
        """
        width, height = original_size
        
        # If no max dimension is set, keep original size
        if not self.max_dimension:
            return original_size
        
        # Find the larger dimension - this is our limiting factor
        if width > height:
            # Landscape orientation
            if width > self.max_dimension:
                ratio = self.max_dimension / width
                new_width = self.max_dimension
                new_height = int(height * ratio)
            else:
                return original_size
        else:
            # Portrait orientation (or square)
            if height > self.max_dimension:
                ratio = self.max_dimension / height
                new_height = self.max_dimension
                new_width = int(width * ratio)
            else:
                return original_size
        
        logger.info(f"Resizing from {width}x{height} to {new_width}x{new_height}")
        return (new_width, new_height)
    
    def _optimize_for_format(self, image: Image.Image, output_format: str) -> Dict:
        """
        Choose the best compression settings for each format.
        Each format has its own personality - we need to speak its language!
        
        Args:
            image: PIL Image object
            output_format: Target format (JPEG, PNG, WEBP, etc.)
            
        Returns:
            Dictionary of optimization parameters for PIL.Image.save()
        """
        if output_format.upper() == 'JPEG':
            return {
                'format': 'JPEG',
                'quality': 85,
                'optimize': True,
                'progressive': True,  # For web-friendly loading
                'subsampling': 0      # Better quality for text/line art
            }
        elif output_format.upper() == 'PNG':
            return {
                'format': 'PNG',
                'optimize': True,
                'compress_level': 9   # Maximum PNG compression
            }
        elif output_format.upper() == 'WEBP':
            return {
                'format': 'WEBP',
                'quality': 85,
                'method': 6,          # Maximum compression effort
                'optimize': True
            }
        else:
            # Default settings for other formats
            return {'format': output_format}
    
    def _apply_intelligent_preprocessing(self, image: Image.Image) -> Image.Image:
        """
        Apply smart preprocessing to improve compression efficiency.
        Think of this as giving the image a spa treatment before compression!
        
        Args:
            image: PIL Image object to preprocess
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert to RGB if necessary (some formats don't play nice with CMYK, etc.)
        if image.mode not in ('RGB', 'L'):
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Apply subtle noise reduction - removes compression artifacts and film grain
        # This is like using a gentle filter to smooth out imperfections
        image = image.filter(ImageFilter.SMOOTH_MORE)
        
        # Slightly enhance contrast to compensate for compression softening
        # We're being proactive here - compression tends to flatten contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)  # Very subtle enhancement
        
        return image
    
    def _compress_iteratively(self, image: Image.Image, output_path: str, 
                            target_size_kb: int) -> bool:
        """
        Iteratively compress until we hit our target size.
        This is like a game of compression limbo - how low can we go?
        
        Args:
            image: PIL Image object to compress
            output_path: Where to save the compressed image
            target_size_kb: Target file size in KB
            
        Returns:
            True if target size was achieved, False otherwise
        """
        # Start with high quality and work our way down
        quality = 95
        min_quality = 20  # We have standards, after all
        
        logger.info(f"Starting iterative compression targeting {target_size_kb}KB")
        
        while quality >= min_quality:
            # Try saving with current quality
            save_params = {
                'format': 'JPEG',
                'quality': quality,
                'optimize': True,
                'progressive': True
            }
            
            # Save to a temporary location first
            temp_path = output_path + '.temp'
            image.save(temp_path, **save_params)
            
            # Check if we've hit our target
            current_size = self._get_file_size_kb(temp_path)
            logger.info(f"Quality {quality}: {current_size}KB")
            
            if current_size <= target_size_kb:
                # Success! Move temp file to final location
                os.rename(temp_path, output_path)
                logger.info(f"Target achieved at quality {quality}")
                return True
            
            # Clean up temp file and try lower quality
            os.remove(temp_path)
            quality -= 5  # Reduce quality in steps of 5
        
        # If we get here, we couldn't reach the target size
        logger.warning(f"Could not achieve target size of {target_size_kb}KB")
        
        # Save with minimum quality as fallback
        save_params = {
            'format': 'JPEG',
            'quality': min_quality,
            'optimize': True,
            'progressive': True
        }
        image.save(output_path, **save_params)
        return False
    
    def compress_image(self, input_path: str, output_path: str, 
                      quality_preset: str = 'high') -> Dict:
        """
        The main compression method - where the magic happens!
        
        This method orchestrates all our compression techniques like a conductor
        leading a symphony orchestra. Each technique plays its part in harmony.
        
        Args:
            input_path: Path to the input image
            output_path: Where to save the compressed image
            quality_preset: Compression quality level
            
        Returns:
            Dictionary with compression statistics and results
        """
        start_time = time.time()
        
        try:
            # Load the image - this is where our journey begins
            logger.info(f"Loading image: {input_path}")
            with Image.open(input_path) as image:
                original_size = image.size
                original_file_size = self._get_file_size_kb(input_path)
                
                logger.info(f"Original image: {original_size[0]}x{original_size[1]}, "
                           f"{original_file_size}KB")
                
                # Step 1: Apply intelligent preprocessing
                # Think of this as preparing the image for its transformation
                image = self._apply_intelligent_preprocessing(image)
                
                # Step 2: Resize if necessary
                # Sometimes smaller is better - like a well-tailored suit
                new_dimensions = self._calculate_new_dimensions(original_size)
                if new_dimensions != original_size:
                    # Use LANCZOS resampling for high-quality resizing
                    # It's like having a professional photographer resize your image
                    image = image.resize(new_dimensions, Image.Resampling.LANCZOS)
                    logger.info(f"Resized to {new_dimensions[0]}x{new_dimensions[1]}")
                
                # Step 3: Determine output format
                # Choose the best format for the job
                output_ext = Path(output_path).suffix.lower()
                if output_ext in ['.jpg', '.jpeg']:
                    output_format = 'JPEG'
                elif output_ext == '.png':
                    output_format = 'PNG'
                elif output_ext == '.webp':
                    output_format = 'WEBP'
                else:
                    # Default to JPEG for maximum compatibility
                    output_format = 'JPEG'
                    output_path = Path(output_path).with_suffix('.jpg')
                
                # Step 4: Apply compression
                if self.target_size_kb:
                    # Size-based compression - we have a specific target
                    success = self._compress_iteratively(image, str(output_path), 
                                                       self.target_size_kb)
                else:
                    # Quality-based compression - we trust our presets
                    quality = self.quality_presets.get(quality_preset, 85)
                    save_params = self._optimize_for_format(image, output_format)
                    
                    # Override quality if it's a quality-based format
                    if 'quality' in save_params:
                        save_params['quality'] = quality
                    
                    image.save(output_path, **save_params)
                    success = True
                
                # Calculate compression statistics
                compressed_file_size = self._get_file_size_kb(output_path)
                compression_ratio = (original_file_size - compressed_file_size) / original_file_size * 100
                processing_time = time.time() - start_time
                
                # Prepare results summary
                results = {
                    'success': success,
                    'original_size_kb': original_file_size,
                    'compressed_size_kb': compressed_file_size,
                    'compression_ratio_percent': round(compression_ratio, 2),
                    'original_dimensions': original_size,
                    'final_dimensions': new_dimensions,
                    'processing_time_seconds': round(processing_time, 2),
                    'output_format': output_format,
                    'quality_preset': quality_preset
                }
                
                logger.info(f"Compression complete: {original_file_size}KB → "
                           f"{compressed_file_size}KB ({compression_ratio:.1f}% reduction)")
                
                return results
                
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    def batch_compress(self, input_folder: str, output_folder: str, 
                      quality_preset: str = 'high') -> List[Dict]:
        """
        Compress multiple images in batch mode.
        Perfect for when you have a folder full of images that need the compression treatment!
        
        Args:
            input_folder: Folder containing images to compress
            output_folder: Where to save compressed images
            quality_preset: Compression quality level
            
        Returns:
            List of compression results for each image
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output folder if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all supported image files
        image_files = []
        for format_name, extensions in self.supported_formats.items():
            for ext in extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to compress")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            # Determine output filename
            output_file = output_path / image_file.name
            
            # Compress the image
            result = self.compress_image(str(image_file), str(output_file), quality_preset)
            result['filename'] = image_file.name
            results.append(result)
            
            # Show progress
            print(f"Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
        
        return results


def main():
    """
    Command-line interface for our image compressor.
    This is where the rubber meets the road!
    """
    parser = argparse.ArgumentParser(
        description="Advanced Image Compressor - Compress images like a pro!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress single image with high quality
  python image_compressor.py -i photo.jpg -o compressed.jpg -q high
  
  # Compress to specific file size
  python image_compressor.py -i photo.jpg -o compressed.jpg -s 500
  
  # Batch compress with size limit
  python image_compressor.py -if photos/ -of compressed/ -d 1920
  
  # Compress with custom settings
  python image_compressor.py -i photo.jpg -o compressed.jpg -q medium -d 1080 -s 300
        """
    )
    
    # Input/output options
    parser.add_argument('-i', '--input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output image file')
    parser.add_argument('-if', '--input-folder', help='Input folder for batch processing')
    parser.add_argument('-of', '--output-folder', help='Output folder for batch processing')
    
    # Compression options
    parser.add_argument('-q', '--quality', 
                       choices=['maximum', 'high', 'medium', 'low', 'minimum'],
                       default='high',
                       help='Compression quality preset (default: high)')
    parser.add_argument('-s', '--size', type=int, 
                       help='Target file size in KB')
    parser.add_argument('-d', '--dimension', type=int,
                       help='Maximum dimension (width or height) in pixels')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not ((args.input and args.output) or (args.input_folder and args.output_folder)):
        parser.error("Either provide -i/-o for single file or -if/-of for batch processing")
    
    # Initialize compressor
    compressor = ImageCompressor(
        target_size_kb=args.size,
        max_dimension=args.dimension,
        preserve_aspect_ratio=True
    )
    
    try:
        if args.input and args.output:
            # Single file compression
            print(f"Compressing {args.input} → {args.output}")
            result = compressor.compress_image(args.input, args.output, args.quality)
            
            if result['success']:
                print(f"✅ Success! Reduced size by {result['compression_ratio_percent']:.1f}%")
                print(f"   Original: {result['original_size_kb']}KB")
                print(f"   Compressed: {result['compressed_size_kb']}KB")
                print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        elif args.input_folder and args.output_folder:
            # Batch processing
            print(f"Batch compressing {args.input_folder} → {args.output_folder}")
            results = compressor.batch_compress(args.input_folder, args.output_folder, args.quality)
            
            # Summary statistics
            successful = [r for r in results if r['success']]
            total_original_size = sum(r['original_size_kb'] for r in successful)
            total_compressed_size = sum(r['compressed_size_kb'] for r in successful)
            total_reduction = (total_original_size - total_compressed_size) / total_original_size * 100
            
            print(f"\n📊 Batch Compression Summary:")
            print(f"   Files processed: {len(results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(results) - len(successful)}")
            print(f"   Total size reduction: {total_reduction:.1f}%")
            print(f"   Space saved: {total_original_size - total_compressed_size:.1f}KB")
            
            # Save detailed results
            results_file = Path(args.output_folder) / 'compression_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Detailed results saved to: {results_file}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Compression interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Main execution error: {str(e)}")


if __name__ == "__main__":
    main()
