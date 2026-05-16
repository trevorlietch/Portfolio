# JavaFX path
$FX_PATH = "C:\Users\Trevor\Downloads\javafx-26_windows-x64_bin-sdk\javafx-sdk-26\lib"
$CLASS_DIR = "classes"

# Compile
Write-Host "Compiling..."
New-Item -ItemType Directory -Force -Path $CLASS_DIR | Out-Null
javac --module-path "$FX_PATH" --add-modules javafx.controls -cp gson-2.10.1.jar -d $CLASS_DIR MainApp.java GuitarTab.java ChordSongEditor.java

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed"
    exit 1
}

Remove-Item -Path "*.class" -Force -ErrorAction SilentlyContinue

# Run
Write-Host "Running..."
java --module-path "$FX_PATH" --add-modules javafx.controls -cp "$CLASS_DIR;gson-2.10.1.jar" MainApp
