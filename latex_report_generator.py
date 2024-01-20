from pylatex import Document, Section, Subsection, Command, Figure
from pylatex.utils import italic, NoEscape
from datetime import datetime
import matplotlib.pyplot as plt


def initialize_report(file_name, code_file_name, author='Sahil Dhawan'):
    current_time = datetime.now()
    time_string = datetime.strftime(current_time, "%Y%m%d_%H%M%S")
    base_dir = 'code_reports\\'
    doc_name = base_dir + 'CODE REPORT-(' + file_name[:-4] + ')-' + time_string
    doc = Document(doc_name)
    doc.preamble.append(Command('title', code_file_name + ': Report for ' + '\"' + file_name + '\"\n'))
    doc.preamble.append(Command('author', author))
    doc.preamble.append(Command('date', datetime.strftime(current_time, "%Y-%m-%d %H:%M:%S")))
    doc.append(NoEscape(r'\maketitle'))
    return doc, doc_name


def section_insert(doc, section_title):
    doc.append(NoEscape(r'\section{'+section_title+'}'))


def plot_insert(doc, caption_text):
    with doc.create(Figure(position='htbp')) as plot:
        plot.add_plot()
        plot.add_caption(caption_text)
    plt.close()


def compile_report(doc, doc_name):
    doc.generate_pdf(doc_name, clean_tex=False, compiler='pdfLaTeX')
    doc.generate_tex(doc_name)

# with doc.create(Section('A section')):
#     doc.append('Some regular text and some ')
#     doc.append(italic('italic text. '))
#
#     with doc.create(Subsection('A subsection')):
#         doc.append('Also some crazy characters: $&#{}')
#
# with doc.create(Section('A second section')):
#     doc.append('Some text.')
