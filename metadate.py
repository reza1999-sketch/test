# 📚 سیستم کامل استخراج اطلاعات کتاب با قدرت API
# 🔍 نسخه پیشرفته - رابط Gradio تاریک و ساده - رفع مشکل نمایش نتایج

# ============================================================================
# 📦 نصب و راه‌اندازی کتابخانه‌ها
# ============================================================================

print("📦 در حال نصب و راه‌اندازی کتابخانه‌ها...")

!pip install transformers torch torchvision torchaudio -q
!pip install pillow pytesseract pdf2image -q
!pip install sentence-transformers scikit-learn -q
!pip install persian-tools arabic-reshaper python-bidi -q
!pip install spacy rapidfuzz -q
!pip install llama-index-core llama-index-llms-openrouter llama-index-readers-file -q
!pip install llama-index-embeddings-huggingface python-dotenv -q
!pip install gradio -q
!python -m spacy download en_core_web_sm -q
!apt-get install -y tesseract-ocr-fas poppler-utils -qq

print("✅ کتابخانه‌ها نصب شدند")

# ============================================================================
# 🔧 Import کتابخانه‌ها
# ============================================================================

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from google.colab import files
import os
from datetime import datetime
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
import torch
import arabic_reshaper
from bidi.algorithm import get_display
import gradio as gr

# تنظیم مسیر Tesseract برای Colab
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

print("🔧 تنظیمات اولیه انجام شد")

# ============================================================================
# 🧠 بخش ۱: مدل‌های هوش مصنوعی با دقت بالا
# ============================================================================

class HighAccuracyAIModels:
    """مدل‌های هوش مصنوعی با دقت فوق العاده"""

    def __init__(self):
        self.setup_high_accuracy_models()

    def setup_high_accuracy_models(self):
        """بارگذاری مدل‌های با دقت بالا"""
        print("🧠 در حال بارگذاری مدل‌ها...")

        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model="HooshvareLab/bert-fa-zwnj-base-ner",
                aggregation_strategy="max",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ مدل NER بارگذاری شد")
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری مدل NER: {e}")
            self.ner_pipeline = None

        try:
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("✅ مدل Embedding بارگذاری شد")
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری مدل Embedding: {e}")
            self.embedding_model = None

# ============================================================================
# 🔍 بخش ۲: OCR با دقت فوق العاده
# ============================================================================

class UltraAccuracyOCREngine:
    """موتور OCR با دقت فوق العاده"""

    def __init__(self, ai_models):
        self.ai_models = ai_models
        self.setup_advanced_ocr()

    def setup_advanced_ocr(self):
        """تنظیمات پیشرفته OCR"""
        self.tesseract_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءةيك۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩ :.,-()',
            r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
        ]

    def extract_text_ultra_accurate(self, input_file, progress_callback=None) -> Tuple[str, Dict[str, Any]]:
        """استخراج متن با دقت فوق العاده"""
        if progress_callback:
            progress_callback(5, "شروع استخراج متن...")

        results = {
            'methods': {},
            'quality_metrics': {}
        }

        try:
            if progress_callback:
                progress_callback(20, "پردازش صفحات...")
            
            tesseract_results = self._advanced_tesseract_extraction(input_file, progress_callback)
            results['methods']['tesseract_advanced'] = tesseract_results

            combined_text = tesseract_results['text']
            results['combined_text'] = combined_text
            results['quality_metrics'] = self._calculate_comprehensive_quality(combined_text)

            if progress_callback:
                progress_callback(80, "استخراج متن کامل شد")
            
            return combined_text, results

        except Exception as e:
            print(f"❌ خطا در استخراج متن: {e}")
            return "", results

    def _advanced_tesseract_extraction(self, input_file, progress_callback=None) -> Dict[str, Any]:
        """استخراج پیشرفته با Tesseract"""
        start_time = datetime.now()

        try:
            if isinstance(input_file, str) and input_file.lower().endswith('.pdf'):
                images = convert_from_path(input_file, first_page=1, last_page=3, dpi=300)
                all_texts = []

                for i, image in enumerate(images):
                    if progress_callback:
                        progress_callback(20 + (i * 15), f"پردازش صفحه {i+1}...")

                    processed_image = self._preprocess_image(image)

                    page_texts = []
                    for config in self.tesseract_configs:
                        try:
                            text = pytesseract.image_to_string(
                                processed_image,
                                lang='fas+eng',
                                config=config
                            )
                            if text.strip():
                                page_texts.append(text)
                        except:
                            continue

                    if page_texts:
                        best_page_text = max(page_texts, key=lambda x: len(x))
                        all_texts.append(best_page_text)

                final_text = '\n'.join(all_texts)

            else:
                if isinstance(input_file, str):
                    image = Image.open(input_file)
                else:
                    image = input_file

                processed_image = self._preprocess_image(image)
                final_text = pytesseract.image_to_string(processed_image, lang='fas+eng')

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'text': final_text,
                'confidence': 0.8,
                'processing_time': processing_time,
                'method': 'tesseract_advanced'
            }

        except Exception as e:
            print(f"⚠️ خطا در Tesseract: {e}")
            return {'text': '', 'confidence': 0, 'processing_time': 0, 'method': 'error'}

    def _preprocess_image(self, image):
        """پیش‌پردازش تصویر"""
        try:
            if image.mode != 'L':
                image = image.convert('L')
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            return image
        except:
            return image

    def _calculate_comprehensive_quality(self, text: str) -> Dict[str, float]:
        """محاسبه کیفیت جامع متن"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        persian_chars = len(re.findall(r'[آ-ی]', text))
        total_chars = len(text)

        return {
            'overall_score': min(persian_chars / max(total_chars, 1) * 2, 1.0),
            'line_count': len(lines),
            'persian_ratio': persian_chars / total_chars if total_chars > 0 else 0,
            'total_chars': total_chars
        }

# ============================================================================
# 🤖 بخش ۳: سیستم RAG ساده‌سازی شده
# ============================================================================

class SimpleRAGSystem:
    """سیستم RAG ساده‌سازی شده"""

    def __init__(self):
        self.api_key = "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"
        self.setup_llm()
        self.knowledge_base = None

    def setup_llm(self):
        """تنظیم مدل زبانی"""
        try:
            from llama_index.llms.openrouter import OpenRouter
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            self.llm = OpenRouter(
                model="meta-llama/llama-3-70b-instruct",
                temperature=0.1,
                max_tokens=2000,
                api_key=self.api_key
            )
            
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            print("✅ مدل زبانی RAG تنظیم شد")

        except Exception as e:
            print(f"❌ خطا در تنظیم مدل RAG: {e}")
            self.llm = None
            self.embed_model = None

    def create_knowledge_base(self, text: str):
        """ایجاد پایگاه دانش"""
        if not self.llm:
            return False

        try:
            from llama_index.core import Document, VectorStoreIndex

            short_text = text[:2000]
            document = Document(text=short_text)
            
            self.knowledge_base = VectorStoreIndex.from_documents(
                [document], 
                embed_model=self.embed_model
            )
            return True

        except Exception as e:
            print(f"❌ خطا در ایجاد پایگاه دانش: {e}")
            return False

# ============================================================================
# 🚀 بخش ۴: استخراج‌کننده متادیتا
# ============================================================================

class APIEnhancedMetadataExtractor:
    """استخراج کننده متادیتا"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def extract_with_api_power(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """استخراج متادیتا با قدرت API"""
        if progress_callback:
            progress_callback(85, "استخراج اطلاعات با هوش مصنوعی...")
        
        if not self.rag_system.llm:
            return self._extract_without_api(text)
        
        try:
            chunk = text[:2000]
            prompt = f"""
            از متن زیر اطلاعات کتاب را استخراج کن:
            {chunk}
            
            اطلاعات مورد نیاز:
            - عنوان کتاب
            - نویسنده/مؤلف
            - مترجم (اگر وجود دارد)
            - ناشر
            - سال انتشار
            - شابک (ISBN)
            - نوبت چاپ
            
            پاسخ را به صورت JSON برگردان.
            """
            
            metadata = self._call_api_for_extraction(prompt)
            return metadata
            
        except Exception as e:
            print(f"❌ خطا در استخراج با API: {e}")
            return self._extract_without_api(text)
    
    def _extract_without_api(self, text: str) -> Dict[str, Any]:
        """استخراج بدون API"""
        metadata = {}
        
        # استخراج عنوان
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:10]:
            if (10 <= len(line) <= 150 and
                len(re.findall(r'[آ-ی]', line)) >= 2 and
                not any(word in line for word in ['نویسنده', 'مؤلف', 'ناشر', 'چاپ', 'شابک'])):
                metadata['title'] = line
                break
        
        # استخراج سایر اطلاعات
        patterns = {
            'author': r'نویسنده\s*[:\-]\s*([^\n]+)',
            'publisher': r'ناشر\s*[:\-]\s*([^\n]+)',
            'publication_year': r'سال\s*انتشار\s*[:\-]\s*([۱۳۴۰-۹]{4})',
            'isbn': r'شابک\s*[:\-]\s*([۰-۹\-–]+)',
            'edition': r'چاپ\s*(\d+)',
            'translator': r'مترجم\s*[:\-]\s*([^\n]+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()
        
        return metadata
    
    def _call_api_for_extraction(self, prompt: str) -> Dict[str, Any]:
        """فراخوانی API"""
        try:
            from llama_index.core import Document
            
            doc = Document(text=prompt)
            query_engine = self.rag_system.knowledge_base.as_query_engine(
                llm=self.rag_system.llm,
                similarity_top_k=2
            )
            
            response = query_engine.query(prompt)
            response_text = str(response).strip()
            
            # استخراج JSON از پاسخ
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            
            return {}
            
        except Exception as e:
            print(f"⚠️ خطا در فراخوانی API: {e}")
            return {}

# ============================================================================
# 📝 بخش ۵: استخراج‌کننده ساده
# ============================================================================

class SimpleBookExtractor:
    """استخراج ساده اطلاعات کتاب"""

    def extract_basic_info(self, text: str) -> Dict[str, Any]:
        """استخراج اطلاعات پایه"""
        results = {}
        lines = text.split('\n')

        # استخراج عنوان
        for line in lines[:10]:
            line = line.strip()
            if (10 <= len(line) <= 150 and
                len(re.findall(r'[آ-ی]', line)) >= 2):
                results['title'] = line
                break

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # استخراج سال
            year_match = re.search(r'۱۳[۷-۹][۰-۹]|۱۴۰[۰-۴]', line)
            if year_match and 'year' not in results:
                results['publication_year'] = year_match.group()

            # استخراج نویسنده
            author_match = re.search(r'نویسنده\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if author_match and 'author' not in results:
                results['author'] = author_match.group(1).strip()

            # استخراج ناشر
            publisher_match = re.search(r'ناشر\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if publisher_match and 'publisher' not in results:
                results['publisher'] = publisher_match.group(1).strip()

            # استخراج مترجم
            translator_match = re.search(r'مترجم\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if translator_match and 'translator' not in results:
                results['translator'] = translator_match.group(1).strip()

            # استخراج شابک
            isbn_match = re.search(r'شابک\s*[:\-]\s*([۰-۹\-–]+)', line, re.IGNORECASE)
            if isbn_match and 'isbn' not in results:
                results['isbn'] = isbn_match.group(1).strip()

            # استخراج نوبت چاپ
            edition_match = re.search(r'چاپ\s*(\d+)', line, re.IGNORECASE)
            if edition_match and 'edition' not in results:
                results['edition'] = edition_match.group(1).strip()

        return results

    def extract_additional_info(self, text: str) -> Dict[str, Any]:
        """استخراج اطلاعات تکمیلی"""
        results = {}

        patterns = {
            'publisher': r'ناشر\s*[:\-]\s*(.+)',
            'isbn': r'شابک\s*[:\-]\s*([۰-۹\-–]+)',
            'translator': r'مترجم\s*[:\-]\s*(.+)',
            'price': r'قیمت\s*[:\-]\s*([۰-۹,]+)',
            'subject': r'موضوع\s*[:\-]\s*(.+)'
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[field] = match.group(1).strip()

        return results

# ============================================================================
# 🔄 بخش ۶: پردازشگر اصلی
# ============================================================================

class UltraAccuracyBookProcessor:
    """پردازشگر اصلی"""
    
    def __init__(self):
        self.ai_models = HighAccuracyAIModels()
        self.ocr_engine = UltraAccuracyOCREngine(self.ai_models)
        self.rag_system = SimpleRAGSystem()
        self.api_extractor = APIEnhancedMetadataExtractor(self.rag_system)
        self.simple_extractor = SimpleBookExtractor()
    
    def process_book_ultra_accurate(self, input_file, progress_callback=None) -> Dict[str, Any]:
        """پردازش کتاب"""
        if progress_callback:
            progress_callback(0, "شروع پردازش...")
        
        start_time = datetime.now()
        
        # استخراج متن
        if progress_callback:
            progress_callback(10, "استخراج متن از فایل...")
        raw_text, ocr_results = self.ocr_engine.extract_text_ultra_accurate(input_file, progress_callback)
        
        if not raw_text or len(raw_text.strip()) < 10:
            return self._create_error_result("متن کافی استخراج نشد", ocr_results, start_time)
        
        # ایجاد پایگاه دانش
        if progress_callback:
            progress_callback(70, "راه‌اندازی هوش مصنوعی...")
        rag_ready = self.rag_system.create_knowledge_base(raw_text)
        
        if rag_ready:
            # استخراج با API
            api_metadata = self.api_extractor.extract_with_api_power(raw_text, progress_callback)
            backup_metadata = self._extract_backup_metadata(raw_text)
            final_metadata = self._final_fusion(api_metadata, backup_metadata, raw_text)
        else:
            # فقط روش معمولی
            if progress_callback:
                progress_callback(75, "استخراج اطلاعات...")
            final_metadata = self._extract_backup_metadata(raw_text)
        
        # ایجاد نتایج نهایی
        if progress_callback:
            progress_callback(95, "ذخیره‌سازی نتایج...")
        results = self._create_final_results(final_metadata, raw_text, ocr_results, 
                                           rag_ready, start_time, input_file)
        
        if progress_callback:
            progress_callback(100, "پردازش کامل شد!")
        
        return results
    
    def _extract_backup_metadata(self, text: str) -> Dict[str, Any]:
        """استخراج پشتیبان"""
        basic = self.simple_extractor.extract_basic_info(text)
        additional = self.simple_extractor.extract_additional_info(text)
        
        # ترکیب اطلاعات پایه و تکمیلی
        combined_basic = basic.copy()
        for key, value in additional.items():
            if key not in combined_basic and value:
                combined_basic[key] = value
        
        return {
            'basic_info': combined_basic,
            'additional_info': additional
        }
    
    def _final_fusion(self, api_metadata: Dict, backup_metadata: Dict, text: str) -> Dict[str, Any]:
        """ترکیب نهایی"""
        # ابتدا اطلاعات پشتیبان را کپی می‌کنیم
        final_basic = backup_metadata.get('basic_info', {}).copy()
        
        # سپس اطلاعات API را اضافه می‌کنیم (در صورت وجود)
        if api_metadata:
            for field, value in api_metadata.items():
                if value and value not in ['یافت نشد', '']:
                    final_basic[field] = value
        
        return {
            'basic_info': final_basic,
            'additional_info': backup_metadata.get('additional_info', {}),
            'api_enhanced': bool(api_metadata and len(api_metadata) > 0)
        }
    
    def _create_final_results(self, metadata: Dict, text: str, ocr_results: Dict,
                            rag_ready: bool, start_time: datetime, input_file) -> Dict[str, Any]:
        """ایجاد نتایج نهایی"""
        
        return {
            'basic_info': metadata.get('basic_info', {}),
            'additional_info': metadata.get('additional_info', {}),
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'ocr_analysis': ocr_results,
            'rag_available': rag_ready,
            'api_enhanced': metadata.get('api_enhanced', False),
            'total_text_length': len(text),
            'file_info': {
                'file_name': os.path.basename(input_file) if isinstance(input_file, str) else 'uploaded_file',
                'file_type': 'PDF' if isinstance(input_file, str) and input_file.lower().endswith('.pdf') else 'Image'
            }
        }
    
    def _create_error_result(self, error: str, ocr_results: Dict, start_time: datetime):
        """ایجاد نتیجه خطا"""
        return {
            'error': error,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'ocr_analysis': ocr_results
        }

print("✅ تمام کلاس‌های اصلی تعریف شدند")

# ============================================================================
# 🎨 بخش ۷: رابط Gradio تاریک و ساده - نمایش کامل نتایج
# ============================================================================

class GradioInterface:
    """رابط کاربری Gradio"""
    
    def __init__(self):
        self.processor = None
        self.current_progress = 0
        self.current_status = "آماده"
        
    def initialize_processor(self):
        """راه‌اندازی پردازشگر"""
        if self.processor is None:
            self.processor = UltraAccuracyBookProcessor()
    
    def update_progress(self, progress, status):
        """به‌روزرسانی پیشرفت"""
        self.current_progress = progress
        self.current_status = status
    
    def process_file(self, file):
        """پردازش فایل"""
        if file is None:
            return "لطفاً یک فایل آپلود کنید", "", "", 0, "آماده"
        
        try:
            self.initialize_processor()
            file_path = file.name

            results = self.processor.process_book_ultra_accurate(
                file_path, 
                progress_callback=self.update_progress
            )
            
            if 'error' in results:
                return f"خطا: {results['error']}", "", "", 0, "خطا"
            
            report = self._generate_complete_report(results)
            download_info = self._save_results(results, file_path)
            json_output = self._generate_json_output(results)
            
            return report, download_info, json_output, 100, "پردازش کامل شد"
            
        except Exception as e:
            return f"خطا در پردازش: {str(e)}", "", "", 0, "خطا"
    
    def _generate_complete_report(self, results: Dict[str, Any]) -> str:
        """تولید گزارش کامل"""
        basic_info = results.get('basic_info', {})
        additional_info = results.get('additional_info', {})
        
        report = "📚 نتایج کامل استخراج اطلاعات کتاب\n"
        report += "=" * 50 + "\n\n"
        
        # اطلاعات اصلی کتاب
        report += "📖 اطلاعات اصلی:\n"
        report += "-" * 20 + "\n"
        
        main_fields = [
            ('title', 'عنوان کتاب'),
            ('author', 'نویسنده/مؤلف'),
            ('translator', 'مترجم'),
            ('publisher', 'ناشر'),
            ('publication_year', 'سال انتشار'),
            ('isbn', 'شابک (ISBN)'),
            ('edition', 'نوبت چاپ')
        ]
        
        for field, title in main_fields:
            value = basic_info.get(field, "یافت نشد")
            report += f"• {title}: {value}\n"
        
        # اطلاعات تکمیلی
        if additional_info:
            report += "\n📋 اطلاعات تکمیلی:\n"
            report += "-" * 20 + "\n"
            for key, value in additional_info.items():
                if value and value != "یافت نشد":
                    display_key = self._translate_key(key)
                    report += f"• {display_key}: {value}\n"
        
        # اطلاعات فنی
        report += "\n🔧 اطلاعات فنی:\n"
        report += "-" * 15 + "\n"
        report += f"• زمان پردازش: {results.get('processing_time', 0):.1f} ثانیه\n"
        report += f"• طول متن استخراج شده: {results.get('total_text_length', 0)} کاراکتر\n"
        report += f"• کیفیت OCR: {results.get('ocr_analysis', {}).get('quality_metrics', {}).get('overall_score', 0):.1%}\n"
        report += f"• سیستم هوش مصنوعی: {'فعال' if results.get('api_enhanced') else 'غیرفعال'}\n"
        
        # اطلاعات فایل
        file_info = results.get('file_info', {})
        report += f"• نام فایل: {file_info.get('file_name', 'نامشخص')}\n"
        report += f"• نوع فایل: {file_info.get('file_type', 'نامشخص')}\n"
        
        report += f"\n🕒 تاریخ پردازش: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def _translate_key(self, key: str) -> str:
        """ترجمه کلیدهای انگلیسی به فارسی"""
        translations = {
            'publisher': 'ناشر',
            'isbn': 'شابک',
            'price': 'قیمت',
            'subject': 'موضوع',
            'translator': 'مترجم',
            'author': 'نویسنده',
            'publication_year': 'سال انتشار',
            'edition': 'نوبت چاپ',
            'title': 'عنوان'
        }
        return translations.get(key, key)
    
    def _save_results(self, results: Dict[str, Any], file_path: str) -> str:
        """ذخیره نتایج"""
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # ذخیره JSON
            json_filename = f"{base_name}_نتایج_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # ذخیره گزارش متنی
            txt_filename = f"{base_name}_گزارش_{timestamp}.txt"
            report_content = self._generate_complete_report(results)
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return f"✅ فایل‌ها با موفقیت ذخیره شدند:\n📄 {json_filename}\n📝 {txt_filename}"
            
        except Exception as e:
            return f"⚠️ خطا در ذخیره‌سازی: {str(e)}"
    
    def _generate_json_output(self, results: Dict[str, Any]) -> str:
        """تولید خروجی JSON"""
        return json.dumps(results, ensure_ascii=False, indent=2)

# ============================================================================
# 🚀 راه‌اندازی رابط Gradio تاریک و ساده
# ============================================================================

def create_dark_simple_interface():
    """ایجاد رابط تاریک و ساده"""
    
    interface = GradioInterface()
    
    # CSS برای رابط تاریک
    dark_css = """
    .gradio-container {
        background: #000000 !important;
        color: #ffffff !important;
        font-family: Arial, sans-serif !important;
    }
    .container {
        background: #000000 !important;
    }
    .panel {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 5px !important;
        padding: 10px !important;
        margin: 5px 0 !important;
    }
    .progress-text {
        color: #00ff00 !important;
        font-weight: bold;
    }
    .dark-button {
        background: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    .dark-button:hover {
        background: #444 !important;
    }
    .dark-input {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    .dark-slider {
        background: #333 !important;
    }
    .success-text {
        color: #00ff00 !important;
    }
    .error-text {
        color: #ff4444 !important;
    }
    """
    
    with gr.Blocks(
        title="استخراج اطلاعات کتاب",
        css=dark_css
    ) as demo:
        
        gr.Markdown(
            """
            <div style='text-align: center; color: white;'>
            <h1>📚 استخراج اطلاعات کتاب</h1>
            <p>آپلود فایل کتاب (PDF یا تصویر) برای استخراج خودکار اطلاعات</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # بخش آپلود فایل
                gr.Markdown("### 📁 آپلود فایل")
                file_input = gr.File(
                    label="",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                    type="filepath",
                    elem_classes="dark-input"
                )
                
                # دکمه پردازش
                process_btn = gr.Button(
                    "🚀 شروع پردازش",
                    variant="primary",
                    elem_classes="dark-button",
                    size="lg"
                )
                
                # بخش پیشرفت
                gr.Markdown("### 📊 پیشرفت")
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="",
                    interactive=False,
                    elem_classes="dark-slider"
                )
                
                progress_text = gr.Textbox(
                    label="وضعیت",
                    value="آماده",
                    interactive=False,
                    elem_classes="dark-input"
                )
            
            with gr.Column(scale=2):
                # بخش نتایج
                gr.Markdown("### 📄 نتایج کامل")
                output_report = gr.Textbox(
                    label="",
                    lines=12,
                    show_copy_button=True,
                    elem_classes="dark-input"
                )
                
                # بخش ذخیره‌سازی
                gr.Markdown("### 💾 ذخیره‌سازی")
                download_info = gr.Textbox(
                    label="",
                    lines=3,
                    interactive=False,
                    elem_classes="dark-input"
                )
                
                # بخش JSON
                gr.Markdown("### 🔧 خروجی فنی (JSON)")
                json_output = gr.Textbox(
                    label="",
                    lines=8,
                    show_copy_button=True,
                    elem_classes="dark-input"
                )
        
        # اتصال رویداد
        process_btn.click(
            fn=interface.process_file,
            inputs=[file_input],
            outputs=[output_report, download_info, json_output, progress_bar, progress_text]
        )
        
        gr.Markdown(
            """
            <div style='text-align: center; color: #888; margin-top: 20px;'>
            <p>سیستم استخراج خودکار اطلاعات کتاب - نسخه ساده و تاریک</p>
            <p>📖 تمام اطلاعات کتاب به صورت کامل نمایش داده می‌شود</p>
            </div>
            """
        )
    
    return demo

def main():
    """برنامه اصلی"""
    print("🎯 راه‌اندازی سیستم استخراج اطلاعات کتاب...")
    print("🖥️  در حال راه‌اندازی رابط کاربری...")
    
    # ایجاد رابط
    demo = create_dark_simple_interface()
    
    print("✅ رابط کاربری آماده است!")
    print("🌐 لینک در حال تولید...")
    
    # راه‌اندازی
    demo.launch(
        share=True,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
