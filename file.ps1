# 1. Cấu hình đường dẫn
$fileSource = "C:\Users\PVL\Desktop\data1.txt"  # Danh sách chuẩn
$fileTarget = "C:\Users\PVL\Desktop\data2.txt"  # Danh sách cần so sánh
$sourceDir  = "C:\Users\PVL\Desktop\InputFolder" # Thư mục chứa file thực tế cần di chuyển
$moveDir    = "C:\Users\PVL\Desktop\Different"   # Thư mục chứa file bị khác biệt

# 2. Tạo thư mục đích nếu chưa có
if (!(Test-Path $moveDir)) { New-Item -ItemType Directory -Path $moveDir }

# 3. Hàm đọc file và loại bỏ phần mở rộng (đuôi)
function Get-NamesWithoutExtension($path) {
    if (Test-Path $path) {
        return Get-Content $path | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_).Trim() }
    }
    return @()
}

# 4. Lấy danh sách tên đã bỏ đuôi
$list1 = Get-NamesWithoutExtension $fileSource
$list2 = Get-NamesWithoutExtension $fileTarget

# 5. Tìm các tên chỉ có trong fileTarget mà không có trong fileSource (Khác biệt)
$diff = Compare-Object -ReferenceObject $list1 -DifferenceObject $list2 | Where-Object { $_.SideIndicator -eq "=>" }

Write-Host "--- Dang xu ly so sanh (Bo duoi file) ---" -ForegroundColor Yellow

# 6. Duyệt các file trong thư mục nguồn
$filesInDir = Get-ChildItem -Path $sourceDir -File

foreach ($file in $filesInDir) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    
    # Nếu tên file (bỏ đuôi) nằm trong danh sách khác biệt ($diff)
    if ($list2 -contains $baseName -and $list1 -notcontains $baseName) {
        Write-Host "Khac biet: Di chuyen $($file.Name)" -ForegroundColor Red
        Move-Item -Path $file.FullName -Destination $moveDir -Force
    }
}

Write-Host "--- Hoan tat! ---" -ForegroundColor Green