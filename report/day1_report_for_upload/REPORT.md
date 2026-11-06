# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** `06/11/2026`

**Runtime Colab:** `T4 GPU`

**Python / PyTorch / Ultralytics:** `Python`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468; `class_name`= cab; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record này nói về cả bức ảnh vì task:image_classification : bài toán phân loại ảnh
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Xác định bởi taxonomy/checkpoint của mô hình
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Vì class_id: định danh chính xác, class_name: tên dể dễ đọc, taxonomy_name :xác định hệ quy chiếu
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Cần định nghĩa các quy tắc cho toàn bộ ảnh, phải xác định trước bài toán phân loại theo đối tượng hay ngữ cảnh toàn ảnh
6. Vì sao không được lấy `score` của model làm ground truth?
   Vì score chỉ thể hiện mức độ tin cậy hoặc điểm dự đoán không chứng minh được đó là dữ liệu thật

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24,498.92 ,348.92]; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Đây là một hình chữ nhật bao quanh người trong ảnh. Góc trên-trái nằm cách mép trái ảnh khoảng 385 pixels và cách mép trên khoảng 69 pixels. Góc dưới-phải nằm tại tọa độ khoảng (499, 349)
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   0.35 : 11 pre , 0.60 : 6 pre
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Khi hạ confidence threshold từ 0.60 xuống 0.20, model giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Bounding box phải bao sát toàn bộ phần vật thể nhìn thấy được, không bao gồm quá nhiều vùng nền xung quanh. Box phải xác định chính xác góc trên-trái và góc dưới-phải. Đối với vật thể bị che khuất hoặc nằm ngoài khung hình, chỉ vẽ box theo phần thực tế nhìn thấy, không tự suy đoán phần bị khuất
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Bồn rửa
   Chọn lí do hỏi Lab Coach vì: cần xác định rõ đối tượng có được phép gán nhãn khi bị cắt mép hay không, cần thống nhất chỉ vẽ phần nhìn thấy hay có quy định khác, cần xác định ngưỡng tối thiểu về vật thể có nhìn thấy hay không

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`= 0.899318; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Bounding Box (Box): Chỉ xác định vùng hình chữ nhật bao quanh vật thể bằng 4 tọa độ (x_min, y_min, x_max, y_max).
   Polygon: Xác định đường biên thực tế của vật thể thông qua nhiều điểm tọa độ (x, y)
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là mã định danh riêng cho từng đối tượng cụ thể trong ảnh
   class_id Xác định loại đối tượng
   tracking_id Theo dõi cùng một đối tượng qua nhiều frame
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Người gán nhãn phải vẽ Polygon bám sát đường viền nhìn thấy được của vật thể, bao gồm toàn bộ phần vật thể quan sát được nhưng không bao gồm nền hoặc vật thể khác. Không tự suy đoán phần bị che khuất; khi không xác định được ranh giới, cần đánh dấu và thực hiện escalation theo quy trình.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Chọn cụm các chiếc bát ở góc dưới bên trái ảnh, trên mặt bàn gỗ. Đây là vị trí có nhiều vật thể nằm sát nhau, khiến đường biên giữa các chiếc bát khó xác định chính xác. Chọn escalation vì nếu không phân biệt được ranh giới giữa các instance thì không nên tự suy đoán khi vẽ mask

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                              | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                               | Annotator làm gì?                                                                                                     | Reviewer xem gì?                                                                       |
| --------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn đại diện cho toàn ảnh: kitchen.                                   | Ảnh có nhiều vật thể khác nhau nhưng chủ đề chính vẫn là nhà bếp                                                                                                 | Gán nhãn kitchen cho toàn ảnh.                                                                                        | Kiểm tra nhãn có phản ánh đúng nội dung tổng thể không.                                |
| Phát hiện vật thể     | Các Bounding Box (x_min, y_min, x_max, y_max) theo pixel và class_id       | Các box ở cụm bát trên bàn phía dưới bên trái khá sát và chồng lấn; người đứng gần lò nướng. Ví dụ JSON có person với score 0.912625 và oven với score 0.686791. | Vẽ box sát vật thể, chọn đúng class, không bỏ sót hoặc gộp các vật thể khác nhau.                                     | Kiểm tra class, vị trí box, độ bao phủ và các vật thể bị bỏ sót.                       |
| Instance segmentation | Mỗi vật thể có instance_id riêng và Polygon gồm các điểm (x, y) theo pixel | Cụm bát trên bàn có nhiều vật thể sát nhau; biên mask dễ bị dính. JSON có các instance kitchen-002, kitchen-003, kitchen-006 đều là bowl.                        | Vẽ mask theo đường biên từng vật thể; không gộp các instance cùng class. Nếu không xác định được biên thì escalation. | Kiểm tra mask có bám sát vật thể, đúng instance, không lấn sang nền hoặc vật thể khác. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   chỉ sử dụng dữ liệu được cung cấp cho mục đích bài lab, không tự ý sao chép, chia sẻ, tải lên hoặc tiết lộ dữ liệu cho bên thứ ba.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   sẽ dừng xử lý, không tự ý chỉnh sửa hoặc tiếp tục gán nhãn và báo ngay cho Lab Coach hoặc Leader phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
