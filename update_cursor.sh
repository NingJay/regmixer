# 设置变量
CURSOR_COMMIT=dc8361355d709f306d5159635a677a571b277bc0

# 重命名文件（文件在 ~/ 目录）
mv ~/cli-alpine-x64.tar.gz ~/cursor-cli.tar.gz
mv ~/vscode-reh-linux-x64.tar.gz ~/cursor-vscode-server.tar.gz

# 创建必要目录
mkdir -p ~/.cursor-server
mkdir -p ~/.cursor-server/cli/servers/Stable-$CURSOR_COMMIT/server/

# 解压 cursor-cli
tar -xzf ~/cursor-cli.tar.gz -C ~/.cursor-server
mv ~/.cursor-server/cursor ~/.cursor-server/cursor-$CURSOR_COMMIT

# 解压 vscode-server
tar -xzf ~/cursor-vscode-server.tar.gz -C ~/.cursor-server/cli/servers/Stable-$CURSOR_COMMIT/server/ --strip-components=1
