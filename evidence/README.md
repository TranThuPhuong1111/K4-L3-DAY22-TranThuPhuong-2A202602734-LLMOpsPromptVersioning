# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Trần Thu Phương — 2A202602734
**Provider:** OpenRouter (`openai/gpt-4o-mini`), embeddings `openai/text-embedding-3-small` qua endpoint `/embeddings` của OpenRouter
**LangSmith project:** `day22-lab`

## Danh sách evidence

| File | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Project `day22-lab`, lọc `name:"rag-query"` → **50 traces** (Bước 1) |
| `02_prompt_hub.png` | Prompt Hub có 2 prompt `tran-thu-phuong-rag-prompt-v1` và `tran-thu-phuong-rag-prompt-v2` |
| `02_ab_routing_log.txt` | Log Bước 2: push và pull cả 2 prompt từ Hub, 50 câu có nhãn `[prompt-v1]`/`[prompt-v2]` |
| `03_ragas_scores.png` | Bảng so sánh V1 vs V2 (4 chỉ số RAGAS) |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json`, kèm số sample hợp lệ của từng chỉ số |
| `03_ragas_run_log.txt` | Log đầy đủ lần chạy Bước 3 (đã bỏ thanh tiến độ) |
| `04_pii_demo_log.txt` | Demo PIIDetector: 7 test case (email, phone, SSN, thẻ tín dụng, nhiều PII, đủ 4 loại, câu sạch) |
| `04_json_demo_log.txt` | Demo JSONFormatter: 6 test case (hợp lệ, fences, nháy đơn, dấu phẩy thừa, lỗi kết hợp, hỏng hẳn) |

Script Bước 4 in cả 2 demo trong một lần chạy nên hai file log `04_*` có cùng nội dung, theo đúng hướng dẫn trong CHECKPOINTS.md.

**Trace công khai** (mở không cần đăng nhập):
- Bước 1, `rag-query`: https://smith.langchain.com/public/878e1b81-3755-4bd3-825b-2c6335eb6e2a/r
- Bước 2, `ab-rag-query`: https://smith.langchain.com/public/b042ae0c-6c9d-4e7c-bd96-659e95db749f/r

Tổng số trace trên LangSmith: 50 `rag-query`, 50 `ab-rag-query`, cộng thêm các run của RAGAS ở Bước 3.

## Kết quả RAGAS (50 cặp QA × 2 prompt, mỗi chỉ số đủ 50/50 sample)

| Metric | V1 (ngắn gọn) | V2 (có cấu trúc) | Cao hơn |
|---|---|---|---|
| faithfulness | 0.9412 | **0.9505** | V2 |
| answer_relevancy | **0.9074** | 0.8852 | V1 |
| context_recall | 1.0000 | 1.0000 | Hòa |
| context_precision | **0.9450** | 0.9417 | V1 (chênh lệch không đáng kể) |

Cả 2 phiên bản đều có **faithfulness ≥ 0.9**.

Độ dài câu trả lời, đo trên 100 câu trả lời của lần chạy này (lấy từ LangSmith):

| | V1 | V2 |
|---|---|---|
| Số từ trung bình | 38.4 | 71.7 |
| Số câu trung bình | 1.9 | 3.2 |

## Phân tích: vì sao V1 và V2 khác điểm

Hai prompt chỉ khác nhau ở phần system message. Retriever giống hệt nhau (FAISS, chunk 500/50, k=3), LLM giống nhau, temperature = 0. Vì vậy chênh lệch về **faithfulness** và **answer_relevancy** đến từ prompt. Chênh lệch về **context_recall/precision** thì không, như giải thích bên dưới.

1. **answer_relevancy: V1 cao hơn (0.907 so với 0.885).** RAGAS sinh ngược vài câu hỏi từ câu trả lời, rồi đo độ tương đồng embedding với câu hỏi gốc. V1 bị giới hạn 2–4 câu nên chỉ trả lời thẳng vào ý chính, câu hỏi sinh ngược gần với câu hỏi gốc. V2 yêu cầu "câu đầu trả lời thẳng, các câu sau bổ sung chi tiết từ context", nên câu trả lời dài gần gấp đôi (71.7 so với 38.4 từ) và kèm thông tin nền (cơ chế, lợi ích, ví dụ). Câu hỏi sinh ngược từ đó rộng hơn và lệch khỏi trọng tâm câu hỏi gốc, nên điểm thấp hơn.

2. **faithfulness: V2 cao hơn một chút (0.951 so với 0.941).** Faithfulness bằng số claim có căn cứ trong context chia cho tổng số claim. V2 có chỉ dẫn rõ "xác định các facts liên quan… Không suy đoán hay thêm kiến thức ngoài context", nên các câu bổ sung đều lấy từ context. Với V1, câu trả lời chỉ khoảng 2 câu, nên mỗi lần mô hình khái quát hóa hoặc diễn đạt lại không sát context sẽ kéo điểm của câu đó xuống mạnh (1 claim không có căn cứ trên 3–4 claim). Câu dài với nhiều claim có căn cứ thì ít bị ảnh hưởng hơn.

3. **context_recall (1.0) và context_precision (~0.94): không phụ thuộc prompt.** Hai chỉ số này chỉ đánh giá các đoạn context được truy xuất so với câu hỏi và đáp án chuẩn. Context của V1 và V2 giống hệt nhau. Recall bằng 1.0 cho thấy knowledge base cùng top-3 retrieval luôn chứa đủ thông tin cho đáp án chuẩn. Chênh lệch 0.003 ở precision chỉ là nhiễu của LLM-judge. Con số này cũng cho thấy chênh lệch dưới khoảng 0.01 giữa hai phiên bản nên xem là chưa có ý nghĩa.

4. **Mức độ tin cậy.** Chênh lệch faithfulness (0.009) cùng cỡ với nhiễu của judge, nên không thể kết luận chắc V2 grounded hơn V1. Lần chạy trước bị lỗi 402 do hết credit OpenRouter: V2 chỉ chấm được khoảng 37% sample và V1 thiếu khoảng 12%. Ở lần đó V1 có faithfulness cao hơn V2 (0.959 so với 0.946). Vì không đủ sample nên không dùng kết quả đó làm evidence, nhưng nó cho thấy thứ tự hai bên có thể đảo. Chênh lệch answer_relevancy (0.022) ổn định hơn và khớp với chênh lệch rõ rệt về độ dài câu trả lời.

**Kết luận:** V1 phù hợp hơn cho hỏi đáp nhanh: trả lời đúng trọng tâm và rẻ hơn khoảng 2 lần về output token. V2 phù hợp khi cần câu trả lời đầy đủ, chi tiết mà vẫn bám context. Nếu triển khai thật, có thể tăng số câu hỏi hoặc lặp lại nhiều lần đánh giá để biết chênh lệch faithfulness có thật hay chỉ là nhiễu, trước khi đưa V2 lên 100% traffic.

## Ghi chú về A/B routing

`get_prompt_version()` dùng `md5(request_id) % 2`, nên cùng một `request_id` luôn ra cùng một phiên bản qua mọi lần chạy (khác với `random` hay `hash()` của Python vốn bị salt theo từng process). Với 50 request `req-0000` … `req-0049`, kết quả chia là V1 = 19, V2 = 31. Hash chia đều về kỳ vọng, nhưng với mẫu nhỏ thì không đảm bảo đúng 50/50.
