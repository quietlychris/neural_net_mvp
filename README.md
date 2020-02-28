### Neural network minimal example in Rust

While dipping my toe in the waters of deep learning, I came across a blog post [__here__](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1) which demonstrated an extremely simple neural net implementation in Python, which was accomplished in only 9 lines of code. I decided it would be a good learning opportunity to re-implement that in Rust. My version, at least the first commit of it, is quite a bit longer, coming in at 64 LoC after being run through `rustfmt`. That said, the Rust autoformat likes to spread things vertically, and I have both quite a bit more whitespace and print debugging included than the version included in the article. The Python version is more more readable, however.

I used `ndarray` for the linear algebra portions, along with heavy use of the `map()` iterator in order to apply functions or operations in an element-wise fashion to the arrays, since I found many of the `ndarray` and `ndarray-linalg` built-ins to be difficult to use, and since high-performance/zero copy hasn't been a hugely important factor for this anyway.

That said, running a naive benchmark on the Rust code with the `--release` flag vs. Python code for 10000 iterations of finding the proper weights, we see:
```
   Rust results    |   Python results
real	0m0.048s   |     0m0.248s
user	0m0.032s   |     0m0.363s
sys	 0m0.016s   |     0m0.219s

 ```
such that the Rust code presents around a 5x speedup.
