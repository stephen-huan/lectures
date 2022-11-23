import sys
input = sys.stdin.readline # fast cin

N = int(input())
s = list(map(int, input().split()))
print(sum(sorted(s)[1::2]))

