Welcome to your new dbt project!

### Using the starter project

Try running the following commands:
- dbt run
- dbt test

### Setup Profile (Quan trọng)

Dự án sử dụng **SQLite**. File cấu hình kết nối nằm ngoài dự án tại `~/.dbt/profiles.yml`.

**1. Mở file cấu hình:**
```bash
# Windows
notepad $env:USERPROFILE\.dbt\profiles.yml

# Linux/macOS
nano ~/.dbt/profiles.yml
```

**2. Cấu hình mẫu (trỏ về shared/data):**
```yaml
sales_forcasting:
  target: dev
  outputs:
    dev:
      type: sqlite
      threads: 1
      database: 'database'
      schema: 'main'
      # Đường dẫn tuyệt đối đến file DB. Folder chứa file PHẢI TỒN TẠI trước.
      # Ví dụ Windows:
      schemas_and_paths:
        main: 'E:\AIO\Project\sales_forecasting_xai\shared\data\dbt\sales.db'
      schema_directory: 'E:\AIO\Project\sales_forecasting_xai\shared\data\dbt'
```

**3. Lưu ý:**
- Dbt sẽ tự tạo file `.db` nếu chưa có khi chạy `dbt run`.
- **BẮT BUỘC:** Thư mục cha (ví dụ: `...\shared\data\dbt`) phải được tạo thủ công trước đó, nếu không sẽ báo lỗi *unable to open database file*.

**4. Kiểm tra kết nối:**
```bash
dbt debug
```


### Resources:
- Learn more about dbt [in the docs](https://docs.getdbt.com/docs/introduction)
- Check out [Discourse](https://discourse.getdbt.com/) for commonly asked questions and answers
- Join the [chat](https://community.getdbt.com/) on Slack for live discussions and support
- Find [dbt events](https://events.getdbt.com) near you
- Check out [the blog](https://blog.getdbt.com/) for the latest news on dbt's development and best practices
