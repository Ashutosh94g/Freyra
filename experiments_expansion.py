from modules.expansion import FreyraExpansion

expansion = FreyraExpansion()

text = 'a handsome man'

for i in range(64):
    print(expansion(text, seed=i))
