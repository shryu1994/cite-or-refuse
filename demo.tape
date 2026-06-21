Output cite-or-refuse-demo.gif
Output cite-or-refuse-demo.mp4

Set Shell "bash"
Set FontSize 18
Set Width 1280
Set Height 820
Set Padding 28
Set TypingSpeed 40ms

Hide
Type "cd /Users/shryu/workspace/cite-or-refuse && export PS1='$ ' && clear"
Enter
Show

Sleep 1s
Type "# cite-or-refuse: answer with a source, or honestly refuse"
Sleep 400ms
Enter
Sleep 700ms

Type "python3 -m cite_or_refuse.cli ask 'How much storage does the Free plan include?'"
Enter
Sleep 2500ms

Type "# docs cover share links, but not passwords -> refuse instead of bluffing"
Enter
Type "python3 -m cite_or_refuse.cli ask 'Can I password-protect a share link?'"
Enter
Sleep 2500ms

Type "# not this assistant's job -> out of scope"
Enter
Type "python3 -m cite_or_refuse.cli ask 'Is Harbor better than Dropbox?'"
Enter
Sleep 2500ms

Type "# refusing is scored as a first-class PASS"
Enter
Type "make eval"
Enter
Sleep 4500ms
