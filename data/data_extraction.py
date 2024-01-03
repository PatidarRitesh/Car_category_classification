import os
import PyPDF2



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def save_text_to_file(text, output_folder, pdf_filename):
    output_filename = os.path.join(output_folder, f"{os.path.splitext(pdf_filename)[0]}.txt")
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

pdf_folder = '/home/patidarritesh/smartsense/pdfs/sedan/'
output_folder = '/home/patidarritesh/smartsense/raw_data/sedan/'

for pdf_filename in os.listdir(pdf_folder):
    if pdf_filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        extracted_text = extract_text_from_pdf(pdf_path)
        save_text_to_file(extracted_text, output_folder, pdf_filename)

print(f"Text extracted from PDFs and saved to {output_folder}")
