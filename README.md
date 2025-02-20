# ai-api-server

### Pytorch를 사용해 만든 연산 모델입니다.
- 'AND', 'OR', 'NAND', 'NOR', 'XOR', 'NOT'을 할 수 있습니다.
- 이항 연산자('AND', 'OR', 'NAND', 'NOR', 'XOR')
  - /predict/{operator}/left/{left}/{right}
- 단항 연산자('NOT')
  - /predict/{operator}/{value}