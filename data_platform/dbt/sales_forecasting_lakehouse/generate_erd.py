import json
from pathlib import Path
from pyhive import hive

def get_node_table_name(node):
    return node.get('alias') or node.get('name')

def generate_dbml(manifest_path, output_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Khởi tạo kết nối pyhive thẳng vào Spark
    try:
        cursor = hive.connect(host='localhost', port=10001).cursor()
        thrift_available = True
        print("[INFO] Đã kết nối thành công PyHive Spark Thrift Server (localhost:10001)")
    except Exception as e:
        cursor = None
        thrift_available = False
        print(f"[WARN] Không thể kết nối Spark Thrift. Sẽ fallback về schema.yml. Lỗi: {e}")

    dbml_lines = []
    
    nodes = {**manifest.get('nodes', {}), **manifest.get('sources', {})}
    node_id_to_table = {}

    for node_id, node in nodes.items():
        if node.get('resource_type') in ['model', 'source']:
            table_name = get_node_table_name(node)
            
            # Lọc bỏ các bảng thừa (staging, intermediate...), chỉ giữ lại Marts
            if not (table_name.startswith('dim_') or table_name.startswith('fact_')):
                continue
                
            schema_name = node.get('schema')
            node_id_to_table[node_id] = table_name
            
            columns_to_write = []
            has_columns = False

            # Phương pháp Engine-Native: Dùng Spark Thrift để móc schema thật
            if thrift_available and schema_name:
                try:
                    cursor.execute(f"DESCRIBE TABLE {schema_name}.{table_name}")
                    results = cursor.fetchall()
                    for row in results:
                        # Kết quả DESCRIBE của Spark: (col_name, data_type, comment)
                        col_name = row[0].strip()
                        col_type = row[1].strip()
                        # Dừng vòng lặp nếu gặp partition info (phần cuối bảng describe iceberg)
                        if col_name.startswith('#') or not col_name:
                            break
                        
                        columns_to_write.append(f"  {col_name} {col_type.replace(' ', '_')}")
                        has_columns = True
                except Exception as e:
                    pass # Nếu table chưa build hoặc ko tồn tại, bỏ qua báo lỗi nhỏ
            
            # Nếu Thrift vô dụng hoặc bảng chưa mọc, lùi về phương pháp cũ (lấy YAML dbt)
            if not has_columns and node.get('columns'): 
                for col_name, col_data in node.get('columns').items():
                    col_type = col_data.get('data_type') or 'varchar'
                    columns_to_write.append(f"  {col_name} {col_type}")
            
            if columns_to_write:
                dbml_lines.append(f"Table {table_name} {{")
                dbml_lines.extend(columns_to_write)
                dbml_lines.append("}")
                dbml_lines.append("")

    # 2. Xử lý Foreign Keys (Ref) dựa trên depended_on.nodes
    # Thay vì Regex chuỗi "ref()", ta dùng chính mảng depends_on.nodes của bài test "relationships"
    # mảng này do chính dbt compile ra, luôn luôn chứa 2 ID: [Parent_Node, Child_Node] (hoặc ngược lại)
    for node_id, node in manifest.get('nodes', {}).items():
        if node.get('resource_type') == 'test':
            test_meta = node.get('test_metadata', {})
            
            if test_meta.get('name') == 'relationships':
                # Ví dụ bài test: relationships_fact_sales_item_daily_date__date__ref_dim_date_
                # "attached_node" luôn trỏ về Child model (cái khai báo test)
                child_node_id = node.get('attached_node') 
                
                # Từ depends_on.nodes, có 2 phần tử (ví dụ: model.A và model.B).
                # Ta bóc lấy cái node nào không phải là child_node_id -> suy ra nó là Parent!
                depends_on_nodes = node.get('depends_on', {}).get('nodes', [])
                parent_node_id = None
                for dep_id in depends_on_nodes:
                    if dep_id != child_node_id:
                        parent_node_id = dep_id
                        break
                
                # Lấy tên cột được so sánh (dbt cấp trong kwargs)
                kwargs = test_meta.get('kwargs', {})
                child_column = node.get('column_name') or kwargs.get('column_name')
                parent_column = kwargs.get('field')

                if child_node_id and parent_node_id and child_column and parent_column:
                    child_table = node_id_to_table.get(child_node_id)
                    parent_table = node_id_to_table.get(parent_node_id)
                    
                    if child_table and parent_table:
                        # Dọn dẹp khoảng trắng/nháy kép nếu có (DBML syntax)
                        child_column = child_column.strip('"\'')
                        parent_column = parent_column.strip('"\'')
                        dbml_lines.append(f"Ref: {child_table}.{child_column} > {parent_table}.{parent_column}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(dbml_lines))
    print(f"Đã xuất sơ đồ ERD (v2.0) thành công ra file: {output_path}")

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    target_dir = current_dir / "target"
    manifest_file = target_dir / "manifest.json"
    output_dbml = current_dir / "lakehouse_erd.dbml"
    
    if not manifest_file.exists():
        print("LỖI: Không tìm thấy manifest.json! Hãy chạy 'dbt docs generate' trước.")
    else:
        generate_dbml(manifest_file, output_dbml)
