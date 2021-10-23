use rust_calc::ast::*;
use rust_calc::error::show_trace;
use rust_calc::interpreter::Interpreter;
use rust_calc::rpn::{RpnCompiler, RpnInterpreter};
use std::io::{self, StdinLock};
use std::io::{BufReader, Lines};

fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn interpreter_main(lines: &mut Lines<BufReader<StdinLock>>) {
    let mut interp = Interpreter::new();
    loop {
        prompt("INTERPRETER> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            let n = match interp.eval(&ast) {
                Ok(n) => n,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };

            println!("{:?}", n);
        } else {
            break;
        }
    }
}

fn rpn_compiler_main(lines: &mut Lines<BufReader<StdinLock>>) {
    let mut compiler = RpnCompiler::new();
    let rpn_interpreter = RpnInterpreter::new();

    loop {
        prompt("RPN COMPILER> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            let rpn = compiler.compile(&ast);
            println!("{}", rpn);

            let n = rpn_interpreter.eval(&rpn);
            println!("{}", n);
        } else {
            break;
        }
    }
}

fn main() {
    use std::io::{stdin, BufRead};

    println!("0: Interpreter");
    println!("1: RPN compiler");

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    if let Some(Ok(line)) = lines.next() {
        if let Ok(1) = line.parse::<u8>() {
            println!("Use RPN Compiler.",);
            rpn_compiler_main(&mut lines);
        } else {
            println!("Use Interpreter.",);
            interpreter_main(&mut lines);
        }
    }
}
