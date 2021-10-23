use crate::ast::*;
use crate::utils::*;
use ::std::fmt;
use std::error::Error as StdError;

/// 評価機を表すデータ型
pub struct Interpreter;

impl Interpreter {
    pub fn new() -> Self {
        Interpreter
    }

    pub fn eval(&mut self, expr: &Ast) -> Result<i64, InterpreterError> {
        use self::AstKind::*;
        match expr.value {
            Num(n) => Ok(n as i64),
            UniOp { ref op, ref e } => {
                let e = self.eval(e)?;
                Ok(self.eval_uniop(op, e))
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                let l = self.eval(l)?;
                let r = self.eval(r)?;
                self.eval_binop(op, l, r)
                    .map_err(|e| InterpreterError::new(e, expr.loc.clone()))
            }
        }
    }

    pub fn eval_uniop(&self, op: &UniOp, n: i64) -> i64 {
        use self::UniOpKind::*;
        match op.value {
            Plus => n,
            Minus => -n,
        }
    }

    pub fn eval_binop(&self, op: &BinOp, l: i64, r: i64) -> Result<i64, InterpreterErrorKind> {
        use self::BinOpKind::*;
        match op.value {
            Add => Ok(l + r),
            Sub => Ok(l - r),
            Mult => Ok(l * r),
            Div => {
                if r == 0 {
                    Err(InterpreterErrorKind::DivisionByZero)
                } else {
                    Ok(l / r)
                }
            }
        }
    }
}
// https://doc.rust-lang.org/std/error/trait.Error.html
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InterpreterErrorKind {
    DivisionByZero,
}

pub type InterpreterError = Annot<InterpreterErrorKind>;

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InterpreterErrorKind::*;
        match self.value {
            DivisionByZero => write!(f, "{}: Division by zero", &self.loc),
        }
    }
}
impl StdError for InterpreterError {}

impl InterpreterError {
    pub fn show_diagnostic(&self, input: &str) {
        eprintln!("{}", self);
        print_annot(input, self.loc.clone());
    }
}
