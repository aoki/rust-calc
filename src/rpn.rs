use crate::ast::*;

/// 逆ポーランド記法へのコンパイラを表すデータ型
pub struct RpnCompiler;

impl RpnCompiler {
    pub fn new() -> Self {
        RpnCompiler
    }

    pub fn compile(&mut self, expr: &Ast) -> String {
        let mut buf = String::new();
        self.compile_inner(expr, &mut buf);
        buf
    }

    fn compile_inner(&mut self, expr: &Ast, buf: &mut String) {
        use self::AstKind::*;
        match expr.value {
            Num(n) => buf.push_str(&n.to_string()),
            UniOp { ref op, ref e } => {
                self.compile_uniop(op, buf);
                self.compile_inner(e, buf)
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                self.compile_inner(l, buf);
                buf.push_str(" ");
                self.compile_inner(r, buf);
                buf.push_str(" ");
                self.compile_binop(op, buf);
            }
        }
    }

    pub fn compile_uniop(&self, op: &UniOp, buf: &mut String) {
        use UniOpKind::*;
        match op.value {
            Plus => buf.push_str("+"),
            Minus => buf.push_str("-"),
        }
    }

    pub fn compile_binop(&self, op: &BinOp, buf: &mut String) {
        use BinOpKind::*;
        match op.value {
            Add => buf.push_str("+"),
            Sub => buf.push_str("-"),
            Mult => buf.push_str("*"),
            Div => buf.push_str("/"),
        }
    }
}

pub struct RpnInterpreter;

impl RpnInterpreter {
    pub fn new() -> Self {
        RpnInterpreter
    }

    pub fn eval(&self, exp: &str) -> f64 {
        let mut stack = Vec::new();

        for token in exp.split_whitespace() {
            if let Ok(num) = token.parse::<f64>() {
                stack.push(num);
            } else {
                match token {
                    "+" => self.apply(&mut stack, |x, y| x + y),
                    "-" => self.apply(&mut stack, |x, y| x - y),
                    "*" => self.apply(&mut stack, |x, y| x * y),
                    "/" => self.apply(&mut stack, |x, y| x / y),
                    _ => panic!("Unknown operator: {}", token),
                }
            }
        }
        stack.pop().expect("Stack underflow")
    }

    pub fn apply<F>(&self, stack: &mut Vec<f64>, fun: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        if let (Some(y), Some(x)) = (stack.pop(), stack.pop()) {
            let z = fun(x, y);
            stack.push(z);
        } else {
            panic!("Stack undeflow");
        }
    }
}
