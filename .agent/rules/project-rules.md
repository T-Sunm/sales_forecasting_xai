---
trigger: always_on
---

# Quy tắc Làm việc với Code

## 1. Chế độ Chỉnh sửa
- **KHÔNG BAO GIỜ** tự động edit/modify bất kỳ file code nào
- **CHỈ edit** khi user yêu cầu rõ ràng bằng từ khóa: "edit", "sửa", "modify", "thay đổi"
- Khi chưa được lệnh edit: chỉ đề xuất, giải thích, review code

## 2. Coding Standards
Khi được phép edit file, tuân thủ:

### Clean Code Principles
- Code ngắn gọn, súc tích, DRY (Don't Repeat Yourself)
- Tên biến/hàm rõ nghĩa, self-documenting
- Functions nhỏ, single responsibility
- Tránh nested logic quá 3 cấp

### Loại bỏ Code thừa
- **KHÔNG** thêm print/console.log debug statements
- **KHÔNG** thêm try-catch/error handling không cần thiết
- **KHÔNG** thêm comments giải thích code đơn giản
- **KHÔNG** thêm validation/checking dư thừa

### Code Style
- Xóa code cũ, không comment out
- Xóa import không dùng
- Xóa variables không dùng
- Giữ code minimal và focused

## 3. Workflow
1. **Đọc yêu cầu kỹ**: Hiểu chính xác user muốn gì
2. **Plan trước**: Suy nghĩ solution tối ưu nhất
3. **Code precisely**: Chỉ viết code cần thiết, không thêm thắt
4. **Verify**: Đảm bảo code chạy đúng logic yêu cầu