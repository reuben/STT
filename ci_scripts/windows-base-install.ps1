$client = New-Object System.Net.WebClient
$shell = New-Object -com shell.application

# Utility function to download a zip file and extract it
function Expand-ZIPFile($file, $destination, $url)
{
    $client.DownloadFile($url, $file)
    $zip = $shell.NameSpace($file)
    foreach($item in $zip.items())
    {
        $shell.Namespace($destination).copyhere($item)
    }
}

# Install Chocolatey
Invoke-Expression ($client.DownloadString('https://chocolatey.org/install.ps1'))

# Install Windows 10 SDK
choco install -y windows-sdk-10.0

# Install Windows 10 SDK version 2004 required by UWP builds
choco install -y windows-sdk-10-version-2004-all

# Install NodeJS v8
choco install -y nodejs --version 12.16.3

# Install git
choco install -y git --version 2.26.2

# Install cURL
choco install -y curl --version 7.75.0.20210225

# Install python2 as well for node-gyp later
choco install -y python2 --version 2.7.16

# Install python3.6
choco install -y python --version 3.6.8

# Install 7zip, since msys2 p7zip behaves erratically
choco install -y 7zip --version 19.0

# Install VisualStudio 2019 Community
choco install -y visualstudio2019community --version 16.5.4.0 --package-parameters "--add Microsoft.VisualStudio.Workload.MSBuildTools;Microsoft.VisualStudio.Component.VC.160 --passive --locale en-US"
choco install -y visualstudio2019buildtools --version 16.5.4.0 --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools;includeRecommended --add Microsoft.VisualStudio.Component.VC.160 --add Microsoft.VisualStudio.Component.NuGet.BuildTools --add Microsoft.VisualStudio.Workload.UniversalBuildTools;includeRecommended --add Microsoft.VisualStudio.Workload.NetCoreBuildTools;includeRecommended --add Microsoft.Net.Component.4.5.TargetingPack --add Microsoft.Net.Component.4.6.TargetingPack --add Microsoft.Net.Component.4.7.TargetingPack --passive --locale en-US"

# vcredist140 required at least for bazel
choco install -y vcredist140 --version 14.16.27027.1

# .Net Framework v4.5.2
choco install -y netfx-4.5.2-devpack --version 4.5.5165101.20180721

# .Net Framework v4.6.2
choco install -y netfx-4.6.2-devpack --version 4.6.01590.20170129

# .Net Framework v4.7.2
choco install -y netfx-4.7.2-devpack --version 4.7.2.20190225

# NuGet
choco install -y nuget.commandline --version 4.9.3

# Carbon for later
choco install -y carbon --version 2.5.0

# Install CUDA v10.1
$client.DownloadFile("https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_425.25_win10.exe", "C:\cuda_10.1.168_425.25_win10.exe")
Start-Process -FilePath "C:\cuda_10.1.168_425.25_win10.exe" -ArgumentList "-s nvcc_10.1 nvprune_10.1 cupti_10.1 gpu_library_advisor_10.1 memcheck_10.1 cublas_dev_10.1 cudart_10.1 cufft_dev_10.1 curand_dev_10.1 cusolver_dev_10.1 cusparse_dev_10.1" -Wait -NoNewWindow

# CuDNN v7.6.0 for CUDA 10.1
#Expand-ZIPFile -File "C:\cudnn-10.1-windows10-x64-v7.6.0.64.zip" -Destination "C:\CUDNN-10.1\" -Url "http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.1-windows10-x64-v7.6.0.64.zip"
md "C:\CUDNN-10.1"
Expand-ZIPFile -File "C:\cudnn-10.1-windows7-x64-v7.6.0.64.zip" -Destination "C:\CUDNN-10.1\" -Url "http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.1-windows7-x64-v7.6.0.64.zip"
cp "C:\CUDNN-10.1\cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\"
cp "C:\CUDNN-10.1\cuda\lib\x64\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\"
cp "C:\CUDNN-10.1\cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\"

# GrantEveryoneSeCreateSymbolicLinkPrivilege
Start-Process "powershell" -ArgumentList "-command `"& {&'Import-Module' Carbon}`"; `"& {&'Grant-Privilege' -Identity Everyone -Privilege SeCreateSymbolicLinkPrivilege}`"" -Wait -NoNewWindow

# Ensure proper PATH setup
[Environment]::SetEnvironmentVariable("PATH", $Env:Path + ";C:\tools\msys64\usr\bin;C:\Python36;C:\Program Files\Git\bin", "Machine")

# Free some space
Start-Process "cmd.exe" -ArgumentList "/c del C:\cuda_*" -Wait -NoNewWindow
Start-Process "cmd.exe" -ArgumentList "/c del C:\cudnn*" -Wait -NoNewWindow
Start-Process "cmd.exe" -ArgumentList "/c del C:\CUDNN*" -Wait -NoNewWindow
