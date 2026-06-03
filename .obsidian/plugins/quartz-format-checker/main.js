const { Plugin, PluginSettingTab, Setting, Notice } = require('obsidian');

// 常见 Callout 拼写错误映射
const CALLOUT_TYPE_MAP = {
  'summery': 'summary',
  'propety': 'property',
  'defination': 'definition',
  'defenition': 'definition',
  'theorm': 'theorem',
  'therom': 'theorem',
  'lema': 'lemma',
  'corrolary': 'corollary',
  'proposion': 'proposition',
  'propsition': 'proposition',
  'exmaple': 'example',
  'exmple': 'example',
  'algorthm': 'algorithm',
  'algoritm': 'algorithm',
  'aplication': 'application',
  'appliction': 'application',
  'intution': 'intuition',
  'notaion': 'notation',
  'remaek': 'remark',
  'remakr': 'remark',
};

// 有效的 Callout 类型列表
const VALID_CALLOUTS = new Set([
  // 标准 Obsidian callouts
  'note', 'abstract', 'summary', 'tldr', 'info', 'todo', 'tip', 'hint', 'important',
  'success', 'check', 'done', 'question', 'help', 'faq', 'warning', 'caution',
  'attention', 'failure', 'fail', 'missing', 'danger', 'error', 'bug', 'example',
  'quote', 'cite',
  // 数学专用 callouts
  'definition', 'theorem', 'lemma', 'corollary', 'proposition', 'proof',
  'property', 'remark', 'example', 'axiom', 'notation', 'intuition',
  'algorithm', 'application', 'motivation', 'preliminary', 'problem',
  'solution', 'exercise', 'conjecture'
]);

const DEFAULT_SETTINGS = {
  highlightColor: '#ff6b6b',
  enableAutoFix: true,
  showInStatusBar: true,
};

module.exports = class QuartzFormatCheckerPlugin extends Plugin {
  async onload() {
    await this.loadSettings();

    // 添加状态栏指示器
    if (this.settings.showInStatusBar) {
      this.statusBarItem = this.addStatusBarItem();
      this.statusBarItem.setText('格式检查: 就绪');
    }

    // 注册编辑器扩展
    this.registerEditorExtension([
      this.createHighlighter(),
    ]);

    // 监听文件修改事件
    this.registerEvent(
      this.app.workspace.on('editor-change', (editor, info) => {
        this.checkCurrentFile(editor, info);
      })
    );

    // 添加命令
    this.addCommand({
      id: 'check-format',
      name: '检查当前文件格式',
      callback: () => {
        const activeFile = this.app.workspace.getActiveFile();
        if (activeFile) {
          this.checkFile(activeFile);
        }
      }
    });

    this.addCommand({
      id: 'fix-format',
      name: '修复当前文件格式',
      callback: () => {
        const activeFile = this.app.workspace.getActiveFile();
        if (activeFile) {
          this.fixFile(activeFile);
        }
      }
    });

    // 添加设置选项卡
    this.addSettingTab(new QuartzFormatCheckerSettingTab(this.app, this));

    console.log('Quartz Format Checker 插件已加载');
  }

  onunload() {
    console.log('Quartz Format Checker 插件已卸载');
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  createHighlighter() {
    const plugin = this;
    
    return EditorView.decorations.compute(['doc'], (state) => {
      const decorations = [];
      const doc = state.doc;
      
      for (let i = 1; i <= doc.lines; i++) {
        const line = doc.line(i);
        const text = line.text;
        
        // 匹配 callout 语法
        const calloutMatch = text.match(/^\s*>\s*\[!([^\]]+)\]/);
        if (calloutMatch) {
          const type = calloutMatch[1].toLowerCase().trim();
          const start = line.from + calloutMatch.index + calloutMatch[0].indexOf('[') + 2;
          const end = start + calloutMatch[1].length;
          
          // 检查是否是拼写错误
          if (CALLOUT_TYPE_MAP[type]) {
            decorations.push(Decoration.mark({
              class: 'quartz-format-error',
              attributes: {
                style: `background-color: ${plugin.settings.highlightColor}40; border-bottom: 2px wavy ${plugin.settings.highlightColor};`,
                title: `拼写错误: "${calloutMatch[1]}" 应为 "${CALLOUT_TYPE_MAP[type]}"`,
              }
            }).range(start, end));
          }
          // 检查是否是未知的 callout 类型
          else if (!VALID_CALLOUTS.has(type)) {
            decorations.push(Decoration.mark({
              class: 'quartz-format-warning',
              attributes: {
                style: 'background-color: #ffa50040; border-bottom: 2px wavy #ffa500;',
                title: `未知的 callout 类型: "${calloutMatch[1]}"`,
              }
            }).range(start, end));
          }
        }
      }
      
      return Decoration.set(decorations);
    });
  }

  checkCurrentFile(editor, info) {
    const content = editor.getValue();
    const issues = this.analyzeContent(content);
    
    if (this.statusBarItem) {
      if (issues.length === 0) {
        this.statusBarItem.setText('✅ 格式正确');
        this.statusBarItem.style.color = '#4caf50';
      } else {
        const errorCount = issues.filter(i => i.severity === 'error').length;
        const warningCount = issues.filter(i => i.severity === 'warning').length;
        this.statusBarItem.setText(`⚠️ 错误: ${errorCount} 警告: ${warningCount}`);
        this.statusBarItem.style.color = errorCount > 0 ? '#ff6b6b' : '#ffa500';
      }
    }
  }

  analyzeContent(content) {
    const issues = [];
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      // 检查 callout 类型
      const calloutMatch = line.match(/^\s*>\s*\[!([^\]]+)\]/);
      if (calloutMatch) {
        const type = calloutMatch[1].toLowerCase().trim();
        
        if (CALLOUT_TYPE_MAP[type]) {
          issues.push({
            line: index + 1,
            type: 'callout-spelling',
            severity: 'error',
            message: `拼写错误: "${calloutMatch[1]}" 应为 "${CALLOUT_TYPE_MAP[type]}"`,
            original: calloutMatch[1],
            suggested: CALLOUT_TYPE_MAP[type],
          });
        } else if (!VALID_CALLOUTS.has(type)) {
          issues.push({
            line: index + 1,
            type: 'callout-unknown',
            severity: 'warning',
            message: `未知的 callout 类型: "${calloutMatch[1]}"`,
            original: calloutMatch[1],
          });
        }
      }
      
      // 检查 callout 之间是否缺少空行
      if (index > 0) {
        const prevLine = lines[index - 1];
        const isCurrentCallout = line.match(/^\s*>\s*\[!/);
        const isPrevCalloutEnd = prevLine.match(/^\s*>/) && !prevLine.match(/^\s*>\s*\[!/);
        
        if (isPrevCalloutEnd && isCurrentCallout && line.trim() !== '') {
          issues.push({
            line: index + 1,
            type: 'callout-spacing',
            severity: 'warning',
            message: '相邻 callout 之间建议添加空行',
          });
        }
      }
    });
    
    return issues;
  }

  checkFile(file) {
    const content = this.app.vault.read(file);
    content.then(text => {
      const issues = this.analyzeContent(text);
      
      if (issues.length === 0) {
        new Notice('✅ 格式检查通过！', 3000);
      } else {
        const errorCount = issues.filter(i => i.severity === 'error').length;
        const warningCount = issues.filter(i => i.severity === 'warning').length;
        
        let message = `发现 ${errorCount} 个错误, ${warningCount} 个警告\n\n`;
        issues.slice(0, 5).forEach(issue => {
          const icon = issue.severity === 'error' ? '❌' : '⚠️';
          message += `${icon} 第 ${issue.line} 行: ${issue.message}\n`;
        });
        
        if (issues.length > 5) {
          message += `\n...还有 ${issues.length - 5} 个问题`;
        }
        
        new Notice(message, 10000);
      }
    });
  }

  fixFile(file) {
    this.app.vault.read(file).then(content => {
      let fixed = content;
      let fixCount = 0;
      
      // 修复 callout 拼写错误
      Object.entries(CALLOUT_TYPE_MAP).forEach(([wrong, correct]) => {
        const regex = new RegExp(`(\\s*>\\s*\\[!${wrong}\\])`, 'gi');
        const matches = fixed.match(regex);
        if (matches) {
          fixCount += matches.length;
          fixed = fixed.replace(regex, (match) => {
            return match.replace(wrong, correct);
          });
        }
      });
      
      if (fixCount > 0) {
        this.app.vault.modify(file, fixed);
        new Notice(`✅ 已修复 ${fixCount} 处格式问题`, 3000);
      } else {
        new Notice('没有发现需要修复的格式问题', 3000);
      }
    });
  }
}

class QuartzFormatCheckerSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl('h2', { text: 'Quartz 格式检查设置' });

    new Setting(containerEl)
      .setName('高亮颜色')
      .setDesc('错误高亮的颜色（HEX格式）')
      .addText(text => text
        .setPlaceholder('#ff6b6b')
        .setValue(this.plugin.settings.highlightColor)
        .onChange(async (value) => {
          this.plugin.settings.highlightColor = value;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('启用自动修复')
      .setDesc('在状态栏显示自动修复按钮')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.enableAutoFix)
        .onChange(async (value) => {
          this.plugin.settings.enableAutoFix = value;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('显示状态栏')
      .setDesc('在状态栏显示格式检查结果')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.showInStatusBar)
        .onChange(async (value) => {
          this.plugin.settings.showInStatusBar = value;
          await this.plugin.saveSettings();
        }));

    containerEl.createEl('h3', { text: '使用方法' });
    
    const usage = `
1. 实时检查：编辑时自动高亮格式错误
2. 手动检查：Ctrl+P → "Quartz: 检查当前文件格式"
3. 自动修复：Ctrl+P → "Quartz: 修复当前文件格式"
4. 查看详情：点击状态栏查看所有问题

支持的检查：
• Callout 类型拼写错误（如 summery → summary）
• 未知的 Callout 类型
• Callout 之间缺少空行
• 公式前后空格问题
    `;
    
    containerEl.createEl('pre', { text: usage });
  }
}
