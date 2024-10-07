from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter  #chia van ban thanh doan nho
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader #load van ban pdf va thu muc
from langchain_community.vectorstores import FAISS #csdl vector de luu tru va truy van nhung (embedding) van ban
from langchain_community.embeddings import GPT4AllEmbeddings #tao nhung (embeddings) tu van ban
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.embeddings")



#Khai bao bien:
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

#Ham 1: tao ra vector DB tu 1 doan text:
def create_db_from_text():
    raw_text = """Bối cảnh lịch sử Việt Nam cuối thế kỷ XIX đầu thế kỷ XX:
Xã hội Việt Nam trước khi Pháp xâm lược là xã hội phong kiến độc lập, nền nông nghiệp lạc hậu, trì trệ. Chính quyền nhà Nguyễn đã thi hành chính sách đối nội, đối ngoại trì trệ, bảo thủ... Vì vậy, đã không phát huy được những thế mạnh của dân tộc, không tạo ra tiềm lực vật chất và tinh thần đủ sức bảo vệ Tổ quốc, chống lại âm mưu xâm lược của chủ nghĩa thực dân phương Tây.
Khi thực dân Pháp xâm lược Việt Nam (1858), xã hội Việt Nam thay đổi về tính chất xã hội từ xã hội phong kiến độc lập trở thành xã hội thuộc địa nửa phong kiến. Trong lòng xã hội thuộc địa, mâu thuẫn mới bao trùm lên mâu thuẫn cũ, nó không thủ tiêu mâu thuẫn cũ mà là cơ sở để duy trì mâu thuẫn cũ, làm cho xã hội Việt Nam càng thêm đen tối. Các phong trào vũ trang kháng chiến chống thực dân Pháp rầm rộ, lan rộng ra cả nước… lãnh đạo họ là các văn thân, sĩ phu mang ý thức hệ phong kiến cuối cùng đều thất bại, điều đó cho thấy sự bất lực của hệ tư tưởng phong kiến trước nhiệm vụ lịch sử của dân tộc.
Các cuộc khai thác thuộc địa của thực dân Pháp khiến cho xã hội nước ta có sự chuyển biến và phân hóa, giai cấp công nhân, tầng lớp tiểu tư sản và tư sản bắt đầu xuất hiện, tạo ra những tiền đề bên trong cho phong trào yêu nước giải phóng dân tộc Việt Nam đầu thế kỷ XX. Cùng thời điểm lịch sử đó, các “tân thư”, “tân báo” và những ảnh hưởng của trào lưu cải cách ở Nhật Bản, Trung Quốc tràn vào Việt Nam, phong trào yêu nước của nhân dân ta chuyển sang xu hướng dân chủ tư sản. Tiêu biểu như phong trào của Phan Bội Châu, Phan Chu Trinh…nhưng tất cả đều thất bại. 
Cùng với phong trào đấu tranh yêu nước của nhân dân, sự ra đời và phong trào đấu tranh của giai cấp mới là giai cấp công nhân Việt Nam sau Chiến tranh thế giới lần thứ nhất, đã làm cho phong trào đấu tranh giải phóng dân tộc ở nước ta thêm những yếu tố mới. Đặc biệt, từ đầu những năm 20 của thế kỷ XX, giai cấp công nhân Việt Nam ra đời và ngày càng lớn mạnh về số lượng và chất lượng. Tuy mới ra đời, nhưng với những tác động của phong trào cách mạng thế giới, phong trào đấu tranh của giai cấp công nhân Việt Nam đã có sự phát triển nhanh chóng và mang bản chất của đấu tranh giai cấp và cách mạng: vì sự nghiệp giải phóng dân tộc, giải phóng xã hội và giải phóng con người. Phong trào yêu nước và phong trào công nhân Việt Nam là cơ sở thực tiễn rất quan trọng cho sự ra đời của tư tưởng Hồ Chí Minh. 
Các phong trào yêu nước thời kỳ này dù dưới ngọn cờ nào cũng thất bại hoặc bị dìm trong bể máu. Cách mạng giải phóng dân tộc ở Việt Nam khủng hoảng và bế tắc về đường lối cứu nước.
Nguyễn Tất Thành sinh ra trong bối cảnh nước mất nhà tan và lớn lên trong phong trào cứu nước của dân tộc, Người đã sớm tìm ra nguyên nhân thất bại của các phong trào giải phóng dân tộc là: các phong trào giải phóng dân tộc đều không gắn với tiến bộ xã hội. Nguyễn Ái Quốc quyết định ra đi tìm đường cứu nước - con đường đưa Nguyễn Ái Quốc đến với tư tưởng Hồ Chí Minh: độc lập dân tộc gắn liền với chủ nghĩa xã hội giải phóng dân tộc phải đi theo con đường mới. Như vậy, sự xuất hiện tư tưởng Hồ Chí Minh là một tất yếu, đáp ứng nhu cầu lịch sử của cách mạng Việt Nam.
"""

    #Chia nho van ban:
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500, #kich thuoc 500 ky tu
        chunk_overlap = 50, 
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)

    #Embedding:
    embedding_model = GPT4AllEmbeddings(model_file='models/all-MiniLM-L6-v2-f16.gguf')


    #Dua vao Faiss vector DB:
    db = FAISS.from_texts(texts = chunks, embedding = embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_files():
    #khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)

    #Embedding:
    embedding_model = GPT4AllEmbeddings(model_file = 'models/all-MiniLM-L6-v2-f16.gguf')
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()
