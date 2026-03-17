<#
    功能：数据集重组脚本
    逻辑：
    1. 删除旧 base 文件夹
    2. 遍历 F = 0.2 到 2.0 (步长 0.1)
    3. 从 obase/0.1N-{F}N/train/{F}N 复制到 base/train/{F}N
    4. 从 obase/0.1N-{F}N/val/{F}N 复制到 base/val/{F}N
#>

# ================= 配置区域 =================
$SourceRoot = ".\20250929Step0.2N"    # 源数据总目录
$DestRoot   = ".\continue5"     # 目标数据总目录
# ===========================================

Write-Host ">>> 开始执行数据集重组任务..." -ForegroundColor Cyan

# 1. 检查并清理旧的 base 文件夹
if (Test-Path -Path $DestRoot) {
    Write-Host "检测到旧的 '$DestRoot' 文件夹，正在删除..." -ForegroundColor Yellow
    Remove-Item -Path $DestRoot -Recurse -Force
}

# 2. 创建新的 base 文件夹结构
Write-Host "正在创建新的文件夹结构..." -ForegroundColor Green
New-Item -Path "$DestRoot\train" -ItemType Directory -Force | Out-Null
New-Item -Path "$DestRoot\val" -ItemType Directory -Force | Out-Null

# 3. 循环处理 0.2 到 2.0
# 为了避免浮点数精度问题，使用整数 2 到 20 进行循环，然后除以 10
for ($i = 2; $i -le 20; $i++) {
    
    # 计算 F 值并格式化为一位小数的字符串 (例如 "0.2", "1.0", "2.0")
    # 注意：这里强制保留一位小数，以匹配 "2.0N" 这种命名习惯
    $F_Val = $i / 10
    $F_Str = "{0:N1}" -f $F_Val 
    
    # 构造文件夹名称
    $FolderName = "${F_Str}N"             # 例如: 0.2N
    $SourceDirName = "0.1N-${FolderName}" # 例如: 0.1N-0.2N

    # 构造完整的源路径
    $SrcTrainPath = Join-Path -Path $SourceRoot -ChildPath "$SourceDirName\train\$FolderName"
    $SrcValPath   = Join-Path -Path $SourceRoot -ChildPath "$SourceDirName\val\$FolderName"

    # 构造完整的目标路径 (复制目的地)
    $DestTrainPath = "$DestRoot\train\"
    $DestValPath   = "$DestRoot\val\"

    # --- 执行复制操作 ---
    
    # 处理 Train
    if (Test-Path -Path $SrcTrainPath) {
        Write-Host "正在复制 Train: $FolderName ..."
        Copy-Item -Path $SrcTrainPath -Destination $DestTrainPath -Recurse -Force
    } else {
        Write-Host "[警告] 未找到源路径: $SrcTrainPath" -ForegroundColor Red
    }

    # 处理 Val
    if (Test-Path -Path $SrcValPath) {
        Write-Host "正在复制 Val  : $FolderName ..."
        Copy-Item -Path $SrcValPath -Destination $DestValPath -Recurse -Force
    } else {
        Write-Host "[警告] 未找到源路径: $SrcValPath" -ForegroundColor Red
    }
}

Write-Host "`n>>> 所有任务执行完毕！" -ForegroundColor Cyan
# 暂停一下以便查看结果 (如果直接双击运行)
if ($Host.Name -eq "ConsoleHost") {
    Read-Host "按 Enter 键退出..."
}