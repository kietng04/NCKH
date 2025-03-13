# Phân loại thể loại phim dựa trên phân tích đánh giá văn bản  

## 1. Giới thiệu và định nghĩa vấn đề  

### 1.1. Giới thiệu  

Trong thời đại công nghệ số, lượng dữ liệu văn bản trên internet ngày càng gia tăng nhanh chóng, đặc biệt là trong lĩnh vực giải trí và truyền thông. Các nền tảng trực tuyến như IMDb, Rotten Tomatoes, Metacritic đã tạo ra một kho tàng dữ liệu khổng lồ dưới dạng đánh giá phim từ khán giả. Những bài đánh giá này không chỉ phản ánh ý kiến cá nhân của người xem mà còn chứa đựng những thông tin quan trọng có thể được khai thác để hỗ trợ các hệ thống phân tích và đề xuất phim. Tuy nhiên, một câu hỏi quan trọng đặt ra là: **Liệu chỉ dựa vào nội dung văn bản của các bài đánh giá, có thể xác định chính xác thể loại của bộ phim hay không?**  

Phân loại thể loại phim từ nội dung đánh giá là một bài toán quan trọng trong **Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP)** và **Học Máy (Machine Learning - ML)**. Thông thường, thể loại của một bộ phim được xác định dựa trên kịch bản, đạo diễn, diễn viên và các yếu tố kỹ thuật. Tuy nhiên, một hướng tiếp cận khác là khai thác thông tin từ các bài đánh giá do khán giả cung cấp để dự đoán thể loại phim. Hướng tiếp cận này có thể mang lại nhiều lợi ích thực tiễn, chẳng hạn như tự động gán nhãn thể loại cho các bộ phim chưa được phân loại hoặc hỗ trợ các hệ thống gợi ý phim theo sở thích cá nhân.  

Bài toán phân loại thể loại phim dựa trên nội dung đánh giá đặt ra nhiều thách thức đáng kể. Ngôn ngữ trong các bài đánh giá có thể rất đa dạng, từ những nhận xét ngắn gọn mang tính chủ quan đến những phân tích chuyên sâu về nội dung phim. Hơn nữa, một bộ phim có thể thuộc nhiều thể loại khác nhau, chẳng hạn như vừa là hành động, vừa là khoa học viễn tưởng hoặc có yếu tố hài hước. Điều này làm cho bài toán trở thành một bài toán **phân loại đa nhãn (multi-label classification)** thay vì phân loại đơn nhãn (single-label classification).  

Do đó, nghiên cứu này đề xuất phương pháp áp dụng các kỹ thuật học máy để giải quyết bài toán phân loại thể loại phim dựa trên nội dung đánh giá. Hai mô hình được lựa chọn để so sánh là **K-Nearest Neighbors (KNN)**, một thuật toán truyền thống trong học máy, và **Multilayer Perceptron (MLP)**, một mô hình mạng nơ-ron nhân tạo. Việc so sánh hai phương pháp này giúp đánh giá xem liệu một thuật toán đơn giản nhưng trực quan như KNN có thể hoạt động tốt hơn so với một mô hình học sâu như MLP trong bối cảnh phân loại thể loại phim dựa trên văn bản.  

---

### 1.2. Định nghĩa vấn đề  

Bài toán đặt ra trong nghiên cứu này là **xác định thể loại của một bộ phim dựa trên nội dung đánh giá do khán giả cung cấp**. Cụ thể, cho một bài đánh giá phim dưới dạng văn bản đầu vào, mô hình cần dự đoán một hoặc nhiều thể loại phù hợp nhất với bộ phim đó.  

Về mặt toán học, bài toán có thể được mô tả như sau:  

- Giả sử tập dữ liệu có **N** bài đánh giá phim, được biểu diễn dưới dạng tập hợp **D** = {\( d_1, d_2, ..., d_N \)}.  
- Mỗi bài đánh giá \( d_i \) là một đoạn văn bản chứa các nhận xét về một bộ phim.  
- Tập thể loại phim được định nghĩa là **C** = {\( c_1, c_2, ..., c_K \)}, trong đó \( K \) là số lượng thể loại phim có trong tập dữ liệu.  
- Mỗi bài đánh giá \( d_i \) có thể được gán một tập con các thể loại từ tập \( C \), tức là nhãn của bài đánh giá được biểu diễn dưới dạng một vector nhị phân **Y** = {\( y_1, y_2, ..., y_N \)}, trong đó \( y_i \in \{0, 1\}^K \).  

Bài toán phân loại đa nhãn này có thể được giải quyết bằng nhiều phương pháp khác nhau, bao gồm các mô hình học máy truyền thống (như KNN, SVM) hoặc các mô hình học sâu (như MLP, LSTM, Transformer). Trong nghiên cứu này, hai phương pháp được lựa chọn để so sánh là **KNN** và **MLP**, với dữ liệu văn bản được biểu diễn bằng **TF-IDF (Term Frequency-Inverse Document Frequency)**.  

Để đánh giá hiệu suất của mô hình, các thước đo quan trọng được sử dụng bao gồm **độ chính xác (accuracy), precision, recall, F1-score và Hamming loss**. Các thước đo này giúp xác định mức độ chính xác của mô hình trong việc dự đoán thể loại phim từ đánh giá văn bản.  



---
## 2. Phân tích dữ liệu  

 ### 2.1. Tổng quan về dữ liệu  

Dữ liệu được sử dụng trong nghiên cứu này bao gồm **7.000 bài đánh giá phim** được thu thập từ tập dữ liệu **Large Movie Review Dataset v1.0** của **Maas et al. (2011)**, một trong những bộ dữ liệu tiêu chuẩn cho các nghiên cứu về xử lý ngôn ngữ tự nhiên (NLP). Tập dữ liệu ban đầu bao gồm **50.000 bài đánh giá**, mỗi bài đánh giá được gán nhãn là tích cực hoặc tiêu cực, tuy nhiên, nghiên cứu này không sử dụng nhãn cảm xúc mà thay vào đó tập trung vào thông tin văn bản để phân loại thể loại phim.  

Mỗi bài đánh giá trong tập dữ liệu có độ dài không cố định, dao động từ vài chục đến hàng nghìn từ. Điều này tạo ra một thách thức trong việc trích xuất thông tin quan trọng, vì một số bài đánh giá có thể chứa nhiều mô tả chi tiết về nội dung phim, trong khi những bài khác chỉ đơn thuần là nhận xét chung chung về cảm nhận cá nhân.  

Bên cạnh đó, dữ liệu về **thể loại phim** được thu thập từ IMDb thông qua **web scraping**, với tổng số **27 thể loại phim khác nhau**. Mỗi bài đánh giá có thể thuộc một hoặc nhiều thể loại, khiến bài toán trở thành một bài toán **phân loại đa nhãn (multi-label classification)** thay vì phân loại đơn nhãn. Một số thể loại phim xuất hiện thường xuyên hơn các thể loại khác, dẫn đến sự mất cân bằng trong dữ liệu huấn luyện.  

### 2.2. Phân phối thể loại phim  

Để hiểu rõ hơn về đặc điểm của dữ liệu, nghiên cứu đã phân tích **tần suất xuất hiện của từng thể loại phim** trong tập dữ liệu. Kết quả phân tích cho thấy các thể loại phổ biến nhất bao gồm **Drama, Action, Comedy và Thriller**, trong khi một số thể loại như **Film-Noir, Musical hoặc War** có số lượng rất ít. Điều này có thể ảnh hưởng đến hiệu suất mô hình, vì các thể loại ít xuất hiện sẽ có ít dữ liệu để huấn luyện, làm giảm khả năng tổng quát hóa của mô hình đối với các thể loại hiếm.  

Ngoài ra, phân tích cũng cho thấy phần lớn các bài đánh giá thuộc về các bộ phim có từ **2 đến 3 thể loại**, trong khi một số ít bài đánh giá chỉ thuộc một thể loại duy nhất. Sự phân bố này đồng nghĩa với việc mô hình không chỉ cần xác định xem một bộ phim có thuộc một thể loại nào đó hay không, mà còn phải dự đoán chính xác **bao nhiêu thể loại mà bộ phim đó thuộc về**.  

### 2.3. Độ dài văn bản và độ đa dạng từ vựng  

Một yếu tố quan trọng khác được phân tích là **độ dài trung bình của các bài đánh giá** và **độ đa dạng từ vựng**. Trung bình, mỗi bài đánh giá có khoảng **250 - 300 từ**, với một số bài đánh giá rất ngắn (dưới 50 từ) và một số bài đánh giá dài hơn 1.000 từ. Điều này có thể ảnh hưởng đến khả năng học của mô hình, vì các bài đánh giá ngắn có thể không cung cấp đủ thông tin để dự đoán thể loại chính xác, trong khi các bài đánh giá quá dài có thể chứa nhiều thông tin không liên quan.  

Độ đa dạng từ vựng cũng được xem xét bằng cách đo **kích thước từ vựng duy nhất (unique vocabulary size)**. Kết quả cho thấy có hơn **50.000 từ khác nhau** trong tập dữ liệu, trong đó có nhiều từ xuất hiện với tần suất rất thấp. Điều này đòi hỏi phải áp dụng các phương pháp tiền xử lý như **lọc từ xuất hiện quá ít** hoặc **sử dụng Word Embeddings** để có thể biểu diễn văn bản hiệu quả hơn. 

----------------------------------------------------------------------------------------

## 3. Chuẩn bị dữ liệu   

### 3.1 Nguồn dữ liệu  

Dữ liệu trong nghiên cứu này được thu thập từ hai nguồn chính. Đầu tiên, tập dữ liệu **Large Movie Review Dataset v1.0** được sử dụng làm nguồn chính cho các bài đánh giá phim. Tập dữ liệu này do **Maas et al. (2011)** cung cấp, bao gồm **50.000 bài đánh giá phim từ IMDb** với nhãn phân loại cảm xúc (tích cực hoặc tiêu cực). Tuy nhiên, nghiên cứu này không sử dụng nhãn cảm xúc, mà thay vào đó tập trung vào việc khai thác nội dung văn bản để phân loại thể loại phim. Do hạn chế về tài nguyên tính toán, chỉ **7.000 bài đánh giá** được chọn lọc để phân tích.  

Bên cạnh đó, thông tin về thể loại phim được thu thập trực tiếp từ **IMDb** thông qua kỹ thuật **web scraping**. Tổng cộng, nghiên cứu thu thập được danh sách **27 thể loại phim**, bao gồm các thể loại phổ biến như hành động (Action), hài hước (Comedy), kinh dị (Horror), khoa học viễn tưởng (Sci-Fi), cùng với một số thể loại ít phổ biến hơn như tài liệu (Documentary) hoặc phim-noir (Film-Noir). Để đảm bảo chất lượng dữ liệu, các thể loại có ít hơn **50 bài đánh giá** bị loại bỏ khỏi tập huấn luyện, nhằm tránh mất cân bằng dữ liệu nghiêm trọng trong quá trình huấn luyện mô hình.  

Một đặc điểm quan trọng của tập dữ liệu này là **một bộ phim có thể thuộc nhiều thể loại khác nhau**, làm cho bài toán trở thành một **bài toán phân loại đa nhãn (multi-label classification)** thay vì phân loại đơn nhãn (single-label classification). Điều này có nghĩa là mỗi bài đánh giá phim có thể có nhiều hơn một nhãn thể loại, làm tăng độ phức tạp của bài toán và yêu cầu mô hình có khả năng dự đoán đồng thời nhiều nhãn.  

### 3.2 Tiền xử lý dữ liệu  

Trong ngôn ngữ tự nhiên, có rất nhiều từ xuất hiện với tần suất cao nhưng không đóng vai trò quan trọng trong việc phân loại, chẳng hạn như "của", "và", "là" trong tiếng Việt hoặc "the", "and", "is" trong tiếng Anh. Những từ này được gọi là **stop words**. Việc loại bỏ stop words giúp giảm số lượng đặc trưng trong văn bản và tập trung vào các từ mang ý nghĩa quan trọng hơn. Trong nghiên cứu này, danh sách stop words được lấy từ thư viện **NLTK (Natural Language Toolkit)**.  


Sau khi văn bản đã được chuẩn hóa, loại bỏ từ dừng và biến đổi về dạng gốc, bước tiếp theo là **biểu diễn văn bản dưới dạng vector số** để mô hình có thể xử lý. Trong nghiên cứu này, phương pháp **Term Frequency-Inverse Document Frequency (TF-IDF)** được sử dụng.

## 4. Phương pháp thuật toán

Nghiên cứu này áp dụng hai mô hình học máy để phân loại thể loại phim dựa trên văn bản đánh giá, bao gồm **Multilayer Perceptron (MLP)** và **K-Nearest Neighbors (KNN)**. Hai mô hình này đại diện cho hai cách tiếp cận khác nhau: một mô hình dựa trên **học sâu (deep learning)** và một mô hình **không tham số (non-parametric learning)**. Việc so sánh hai phương pháp này giúp đánh giá xem liệu mạng nơ-ron nhân tạo có thể vượt trội hơn một thuật toán đơn giản nhưng hiệu quả như KNN hay không.  

### 4.1. Multilayer Perceptron (MLP)  

**Multilayer Perceptron (MLP)** là một dạng mạng nơ-ron nhân tạo có cấu trúc nhiều lớp, bao gồm **tầng đầu vào (input layer), tầng ẩn (hidden layers) và tầng đầu ra (output layer)**. Trong mô hình này, các đầu vào là các vector **TF-IDF** biểu diễn văn bản đánh giá phim. Tầng ẩn của MLP đóng vai trò quan trọng trong việc trích xuất đặc trưng, với mỗi nơ-ron kết nối đầy đủ với nơ-ron của tầng tiếp theo.  

Trong quá trình huấn luyện, MLP sử dụng **thuật toán lan truyền ngược (Backpropagation)** để cập nhật trọng số theo hướng giảm thiểu sai số dự đoán. Hàm mất mát được sử dụng là **binary cross-entropy**, do đây là bài toán phân loại đa nhãn, trong đó mỗi thể loại phim là một lớp nhị phân riêng biệt. Để tối ưu hóa mô hình, thuật toán **Adam Optimizer** được sử dụng do khả năng điều chỉnh tốc độ học và tối ưu nhanh chóng hơn so với SGD thông thường.  

Một trong những thách thức lớn của MLP là **quá khớp (overfitting)** do số lượng tham số lớn. Để giải quyết vấn đề này, nghiên cứu đã áp dụng hai kỹ thuật phổ biến là **Dropout** và **Batch Normalization**. Dropout giúp giảm bớt sự phụ thuộc của mô hình vào một số nơ-ron nhất định, bằng cách vô hiệu hóa ngẫu nhiên một số nơ-ron trong quá trình huấn luyện. Batch Normalization giúp chuẩn hóa đầu vào của mỗi lớp, ổn định quá trình học và giảm thiểu sự thay đổi của dữ liệu đầu vào giữa các batch.  

### 4.2. K-Nearest Neighbors (KNN)  

**K-Nearest Neighbors (KNN)** là một thuật toán phân loại dựa trên nguyên tắc tìm các điểm dữ liệu gần nhất trong không gian đặc trưng. Mô hình KNN không yêu cầu giai đoạn huấn luyện trước, mà thay vào đó, khi có một bài đánh giá mới cần phân loại, thuật toán sẽ tính khoảng cách giữa bài đánh giá đó với tất cả các bài đánh giá trong tập dữ liệu huấn luyện.  

Trong nghiên cứu này, khoảng cách giữa các bài đánh giá được đo bằng **khoảng cách Euclidean**, một phương pháp phổ biến để tính toán sự tương đồng giữa hai vector. Khi đã xác định được **K láng giềng gần nhất**, mô hình sẽ sử dụng phương pháp **bỏ phiếu đa số (majority voting)** để quyết định thể loại của bài đánh giá đó.  

Một ưu điểm lớn của KNN là tính đơn giản và không yêu cầu tham số huấn luyện phức tạp. Tuy nhiên, nhược điểm của thuật toán này là **tốc độ chậm khi truy vấn**, do mỗi lần phân loại mới đều yêu cầu tính toán khoảng cách với tất cả các điểm dữ liệu trong tập huấn luyện. Để tối ưu hóa thuật toán, nghiên cứu này sử dụng **kỹ thuật KD-Tree**, giúp tăng tốc quá trình tìm kiếm láng giềng gần nhất trong không gian nhiều chiều.  

Việc lựa chọn **K - số lượng hàng xóm** có ảnh hưởng quan trọng đến hiệu suất của KNN. Nếu K quá nhỏ, mô hình có thể nhạy cảm với nhiễu trong dữ liệu; nếu K quá lớn, mô hình có thể mất đi khả năng phân biệt giữa các thể loại khác nhau. Trong nghiên cứu này, giá trị **K được tối ưu hóa thông qua thử nghiệm**, nhằm đảm bảo độ chính xác cao nhất.  

---
## 5. Cải thiện kết quả  

Mặc dù mô hình K-Nearest Neighbors (KNN) hoạt động tốt hơn Multilayer Perceptron (MLP) trong bài toán phân loại thể loại phim từ văn bản đánh giá, nhưng kết quả thực nghiệm cho thấy cả hai mô hình vẫn chưa đạt độ chính xác cao như mong đợi. Nguyên nhân có thể xuất phát từ nhiều yếu tố, bao gồm cách biểu diễn đặc trưng văn bản, tính chất của tập dữ liệu, khả năng tổng quát hóa của mô hình, và phương pháp đánh giá hiệu suất. Do đó, để nâng cao độ chính xác của hệ thống, một số chiến lược cải tiến có thể được áp dụng, bao gồm việc sử dụng phương pháp biểu diễn văn bản tiên tiến hơn, áp dụng các mô hình học sâu hiện đại, mở rộng tập dữ liệu huấn luyện, điều chỉnh siêu tham số và sử dụng các kỹ thuật học tập tổ hợp.  

Một trong những cải tiến quan trọng là thay thế phương pháp biểu diễn văn bản TF-IDF bằng các kỹ thuật nhúng từ (Word Embeddings), chẳng hạn như Word2Vec, GloVe hoặc BERT. TF-IDF chỉ đơn thuần dựa trên tần suất xuất hiện của từ trong văn bản mà không nắm bắt được ngữ cảnh và mối quan hệ ngữ nghĩa giữa các từ. Trong khi đó, Word Embeddings biểu diễn từ dưới dạng vector trong không gian nhiều chiều, cho phép mô hình hiểu được sự tương đồng giữa các từ dựa trên bối cảnh sử dụng. Việc áp dụng Word2Vec hoặc GloVe có thể giúp mô hình học được các đặc trưng ngữ nghĩa quan trọng, từ đó cải thiện độ chính xác trong dự đoán thể loại phim. Đặc biệt, việc sử dụng BERT hoặc các biến thể của Transformer có thể giúp mô hình học được mối quan hệ giữa các từ trong toàn bộ câu thay vì chỉ xem xét từng từ một cách độc lập.  

Ngoài ra, việc áp dụng các mô hình học sâu tiên tiến hơn cũng là một hướng đi quan trọng để nâng cao hiệu suất phân loại. Mạng nơ-ron MLP có cấu trúc đơn giản, không có khả năng nắm bắt được ngữ cảnh trong văn bản dài. Do đó, việc sử dụng Long Short-Term Memory (LSTM) hoặc Bidirectional LSTM (BiLSTM) có thể giúp mô hình học được thông tin tuần tự trong câu, đặc biệt phù hợp với dữ liệu văn bản có tính chất ngữ nghĩa phức tạp như các bài đánh giá phim. Bên cạnh đó, mạng nơ-ron tích chập (CNN) có thể được thử nghiệm để học các đặc trưng quan trọng trong văn bản, đặc biệt là khi kết hợp với Word Embeddings. Hơn nữa, các mô hình Transformer như BERT, RoBERTa hoặc T5 có thể giúp cải thiện đáng kể hiệu suất phân loại, nhờ vào cơ chế tự chú ý (self-attention) giúp mô hình hiểu được mối quan hệ giữa các từ trong toàn bộ văn bản thay vì chỉ xét ngữ cảnh cục bộ.  

Một yếu tố quan trọng khác ảnh hưởng đến độ chính xác của mô hình là kích thước và sự cân bằng của tập dữ liệu. Trong nghiên cứu này, tập dữ liệu có sự mất cân bằng giữa các thể loại phim, với một số thể loại có nhiều bài đánh giá hơn các thể loại khác. Điều này có thể khiến mô hình bị thiên vị đối với các thể loại phổ biến và kém chính xác đối với các thể loại hiếm. Để khắc phục vấn đề này, có thể áp dụng các phương pháp như **oversampling** (tăng số lượng mẫu từ các lớp ít xuất hiện) hoặc **undersampling** (giảm số lượng mẫu từ các lớp phổ biến hơn) để cân bằng tập dữ liệu. Ngoài ra, việc thu thập thêm dữ liệu từ các nguồn khác như Rotten Tomatoes, Metacritic hoặc Letterboxd có thể giúp cải thiện khả năng tổng quát hóa của mô hình bằng cách cung cấp thêm các mẫu dữ liệu đa dạng hơn.  

Bên cạnh đó, điều chỉnh siêu tham số cũng là một bước quan trọng để tối ưu hóa mô hình. Trong bài toán này, các siêu tham số như số lượng nơ-ron trong tầng ẩn, tỷ lệ học (learning rate), số lượng tầng ẩn trong MLP hoặc số lượng láng giềng trong KNN đều có ảnh hưởng lớn đến hiệu suất của mô hình. Việc sử dụng các phương pháp tối ưu hóa như Grid Search, Random Search hoặc Bayesian Optimization có thể giúp tìm ra các cấu hình siêu tham số phù hợp nhất.  

Cuối cùng, một phương pháp hiệu quả để nâng cao độ chính xác của mô hình là sử dụng kỹ thuật học tập tổ hợp (Ensemble Learning). Các phương pháp như Bagging (Bootstrap Aggregating), Boosting (Gradient Boosting, AdaBoost, XGBoost) và Stacking có thể giúp kết hợp sức mạnh của nhiều mô hình để đạt hiệu suất tốt hơn. Đặc biệt, việc sử dụng mô hình meta-classifier để tổng hợp kết quả từ nhiều mô hình có thể giúp cải thiện độ chính xác tổng thể và giảm thiểu sai số dự đoán.  

---

## 6. Trình bày kết quả  

Sau khi thực hiện huấn luyện và đánh giá mô hình trên tập dữ liệu IMDb, kết quả thực nghiệm được tổng hợp để so sánh hiệu suất giữa các mô hình. Bảng dưới đây trình bày các chỉ số đánh giá chính, bao gồm độ chính xác (Accuracy), Precision, Recall, Hamming Loss và F1-score cho hai mô hình KNN và MLP.  

| Mô hình  | Accuracy | Precision_micro | Recall_micro | Hamming loss | F1-score |
|----------|---------|----------------|-------------|--------------|----------|
| **MLP**  | 37%    | 81.6%          | 59.7%       | 0.051        | 69.2%    |
| **KNN**  | **55.4%**  | 80.7%          | **67.7%**   | **0.047**    | **73.4%** |

Kết quả thực nghiệm cho thấy mô hình **KNN hoạt động tốt hơn MLP**, với độ chính xác đạt **55.4%**, cao hơn đáng kể so với **37%** của MLP. Ngoài ra, **Hamming Loss thấp hơn ở KNN**, cho thấy mô hình này mắc ít lỗi hơn trong việc dự đoán thể loại phim. Điều này có thể do KNN có khả năng hoạt động tốt với dữ liệu có số lượng mẫu vừa phải, trong khi MLP có thể chưa tận dụng được toàn bộ tiềm năng của mạng nơ-ron do kích thước tập dữ liệu huấn luyện còn hạn chế.  

Bên cạnh bảng kết quả, việc trực quan hóa dữ liệu thông qua các biểu đồ cũng giúp dễ dàng phân tích hiệu suất của mô hình. Biểu đồ cột (Bar Chart) có thể được sử dụng để so sánh độ chính xác và F1-score giữa các mô hình, trong khi biểu đồ đường (Line Chart) có thể hiển thị sự thay đổi của độ chính xác theo số lượng dữ liệu huấn luyện. Ngoài ra, ma trận nhầm lẫn (Confusion Matrix) có thể giúp xác định các thể loại phim mà mô hình dễ mắc lỗi nhất, từ đó đưa ra các chiến lược cải tiến phù hợp.  

Tóm lại, kết quả nghiên cứu chỉ ra rằng việc phân loại thể loại phim từ văn bản đánh giá là một bài toán phức tạp, đòi hỏi mô hình phải có khả năng hiểu được ngữ nghĩa và ngữ cảnh của văn bản. Mặc dù KNN đã cho thấy hiệu suất tốt hơn so với MLP, nhưng vẫn còn nhiều hướng nghiên cứu tiềm năng có thể giúp cải thiện kết quả, bao gồm việc sử dụng các mô hình học sâu tiên tiến, mở rộng tập dữ liệu huấn luyện và áp dụng các phương pháp học tập tổ hợp.  





