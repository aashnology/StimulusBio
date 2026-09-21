[1mdiff --git a/tests/test_io_metadata.py b/tests/test_io_metadata.py[m
[1mindex 1a81cba..f7f7eb9 100644[m
[1m--- a/tests/test_io_metadata.py[m
[1m+++ b/tests/test_io_metadata.py[m
[36m@@ -235,14 +235,9 @@[m [mdef test_metadata_loader_preserves_row_order_and_values(tmp_path: Path):[m
         "batch_1",[m
         "batch_1",[m
     ][m
[32m+[m[32mdef test_directory_path_raises_value_error(tmp_path: Path):[m
[32m+[m[32m    directory = tmp_path / "metadata_directory"[m
[32m+[m[32m    directory.mkdir()[m
 [m
[31m-[m
[31m-def test_metadata_loader_reports_missing_file_consistently(tmp_path: Path):[m
[31m-    file_path = tmp_path / "missing.csv"[m
[31m-[m
[31m-    with pytest.raises([m
[31m-        FileNotFoundError,[m
[31m-        match="Metadata file not found",[m
[31m-    ):[m
[31m-        load_metadata(file_path)[m
[31m-[m
[32m+[m[32m    with pytest.raises(ValueError, match="path is not a file"):[m
[32m+[m[32m        load_metadata(directory)[m
