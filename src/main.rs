fn show_trace<E: StdError>(e: E) {
    eprintln!("{}", e);
    let mut source = e.source();
    while let Some(e) = source {
        eprintln!("caused by {}", e);
        source = e.source();
    }
}

fn print_annot(input: &str, loc: Loc) {
    eprintln!("{}", input);
    eprintln!("{}{}", " ".repeat(loc.0), "^".repeat(loc.1 - loc.0));
}

impl Error {
    fn show_daiagnostic(&self, input: &str) {
        use self::Error::*;
        use self::ParseError as P;

        let (e, loc): (&StdError, Loc) = match self {
            Lexer(e) => (e, e.loc.clone()),
            Parser(e) => {
                let loc = match e {
                    P::UnexpectedToken(Token { loc, .. })
                    | P::NotExpression(Token { loc, .. })
                    | P::NotOperator(Token { loc, .. })
                    | P::UncloseOpenParen(Token { loc, .. }) => loc.clone(),
                    P::RedundantExpression(Token { loc, .. }) => Loc(loc.0, input.len()),
                    P::Eof => Loc(input.len(), input.len() + 1),
                };
                (e, loc)
            }
        };
        eprintln!("{}", e);
        print_annot(input, loc);
    }
}

use ::std::fmt;

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TokenKind::*;
        match self {
            Number(n) => n.fmt(f),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            Asterisk => write!(f, "*"),
            Slash => write!(f, "/"),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
        }
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LexErrorKind::*;
        let loc = &self.loc;
        match self.value {
            InvalidChar(c) => write!(f, "{}: invalid char '{}'", loc, c),
            Eof => write!(f, "End of file"),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ParseError::*;
        match self {
            UnexpectedToken(tok) => write!(f, "{}: '{}' is not expected", tok.loc, tok.value),
            NotExpression(tok) => write!(
                f,
                "{}: '{}' is not a start of expression",
                tok.loc, tok.value
            ),
            NotOperator(tok) => write!(f, "{}: '{}' is not an operator", tok.loc, tok.value),
            UncloseOpenParen(tok) => write!(f, "{}: '{}' is not close", tok.loc, tok.value),
            RedundantExpression(tok) => write!(
                f,
                "{}: expression after '{}' is redundant",
                tok.loc, tok.value
            ),
            Eof => write!(f, "End of file"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parse error")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Error {
    Lexer(LexError),
    Parser(ParseError),
}

impl From<LexError> for Error {
    fn from(e: LexError) -> Self {
        Error::Lexer(e)
    }
}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parser(e)
    }
}

use std::error::Error as StdError;
impl StdError for LexError {}
impl StdError for ParseError {}
impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        use self::Error::*;
        match self {
            Lexer(lex) => Some(lex),
            Parser(parse) => Some(parse),
        }
    }
}

/// ATOM = UNUMBER | "(", EXPR3, ")" ;
fn parse_atom<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    tokens
        .next()
        .ok_or(ParseError::Eof)
        .and_then(|tok| match tok.value {
            // UNUMBER
            TokenKind::Number(n) => Ok(Ast::num(n, tok.loc)),
            // | "(" EXPR3 ")"
            TokenKind::LParen => {
                let e = parse_expr3(tokens)?;
                match tokens.next() {
                    Some(Token {
                        value: TokenKind::RParen,
                        .. // ..で値の残りの部分を無視する https://doc.rust-jp.rs/book-ja/ch18-03-pattern-syntax.html
                    }) => Ok(e),
                    Some(t) => {
                        Err(ParseError::RedundantExpression(t))
                    },
                    _ => Err(ParseError::UncloseOpenParen(tok)),
                }
            }
            _ => Err(ParseError::NotExpression(tok)),
        })
}

/// EXPR1 = ("+" | "-"), ATOM | ATOM
fn parse_expr1<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    match tokens.peek().map(|tok| tok.value) {
        Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
            // ("+" | "-")
            let op = match tokens.next() {
                Some(Token {
                    value: TokenKind::Plus,
                    loc,
                }) => UniOp::plus(loc),
                Some(Token {
                    value: TokenKind::Minus,
                    loc,
                }) => UniOp::minus(loc),
                _ => unreachable!(),
            };

            // ATOM
            let e = parse_atom(tokens)?;
            let loc = op.loc.merge(&e.loc);
            Ok(Ast::uniop(op, e, loc))
        }
        // ATOM
        _ => parse_atom(tokens),
    }
}

fn parse_left_binop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<BinOp, ParseError>,
) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    let mut e = subexpr_parser(tokens)?;
    loop {
        match tokens.peek() {
            Some(_) => {
                let op = match op_parser(tokens) {
                    Ok(op) => op,
                    Err(_) => break,
                };

                let r = subexpr_parser(tokens)?;
                let loc = e.loc.merge(&r.loc);
                e = Ast::binop(op, e, r, loc);
            }
            _ => break,
        }
    }
    Ok(e)
}

/// EXPR2 = EXPR2, ("*" | "/"), EXPR1 | EXPR2 ;
/// EXPR2 = EXPR1 EXPR2_Loop
/// EXPR2_Loop = ("*" | "/"), EXPR1, EXPR3_Loop | ε ;
fn parse_expr2<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    fn parse_expr1_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Asterisk => Ok(BinOp::mult(tok.loc.clone())),
                TokenKind::Slash => Ok(BinOp::div(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }
    parse_left_binop(tokens, parse_expr1, parse_expr1_op)
}

/// EXPR3 = EXPR3, ("+" | "-"), EXPR2, | EXPR2 ;
/// 左再帰の除去
/// EXPR3 = EXPR2 EXPR3_Loop
/// EXPR3_Loop = ("+" | "-") EXPR2 EXPR3_Loop | ε
/// 意味的には
/// EXPR2 loop{("+" | "-") EXPR2 | break}
/// となる
fn parse_expr3<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    fn parse_expr3_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
    where
        Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Plus => Ok(BinOp::add(tok.loc.clone())),
                TokenKind::Minus => Ok(BinOp::sub(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, parse_expr2, parse_expr3_op)
}

/// EXPR = EXPR3 ;
fn parse_expr<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    parse_expr3(tokens)
}

fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
    let mut tokens = tokens.into_iter().peekable();
    let ret = parse_expr(&mut tokens)?;
    match tokens.next() {
        Some(tok) => Err(ParseError::RedundantExpression(tok)),
        None => Ok(ret),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParseError {
    /// 予期しないトークンが来た
    UnexpectedToken(Token),
    /// 式を期待していたのに式でないものが来た
    NotExpression(Token),
    /// 演算子を期待していたのに演算子でないものが来た
    NotOperator(Token),
    /// 括弧が閉じられていない
    UncloseOpenParen(Token),
    /// 式の解析が終わったのにまだトークンが残っている
    RedundantExpression(Token),
    /// パース途中で入力が終わった
    Eof,
}

// Binary Operator data structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BinOpKind {
    /// 加算
    Add,
    /// 減算
    Sub,
    /// 乗算
    Mult,
    /// 除算
    Div,
}

type BinOp = Annot<BinOpKind>;

impl BinOp {
    fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::Add, loc)
    }
    fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::Sub, loc)
    }
    fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::Mult, loc)
    }
    fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::Div, loc)
    }
}

/// Unray Operator data structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum UniOpKind {
    Plus,
    Minus,
}

type UniOp = Annot<UniOpKind>;

impl UniOp {
    fn plus(loc: Loc) -> Self {
        Self::new(UniOpKind::Plus, loc)
    }
    fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::Minus, loc)
    }
}

/// AST data structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AstKind {
    /// Number
    Num(u64),
    /// Unray operation
    UniOp { op: UniOp, e: Box<Ast> },
    /// Binary operation
    BinOp { op: BinOp, l: Box<Ast>, r: Box<Ast> },
}
type Ast = Annot<AstKind>;

impl Ast {
    fn num(n: u64, loc: Loc) -> Self {
        Self::new(AstKind::Num(n), loc)
    }

    fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }

    fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }
}

use std::str::FromStr;
impl FromStr for Ast {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = lex(s)?;
        let ast = parse(tokens)?;
        Ok(ast)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

type LexError = Annot<LexErrorKind>;

impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        Self::new(LexErrorKind::InvalidChar(c), loc)
    }
    fn eof(loc: Loc) -> Self {
        Self::new(LexErrorKind::Eof, loc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TokenKind {
    /// [0-9][0-9]*
    Number(u64),
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Asterisk,
    /// /
    Slash,
    /// (
    LParen,
    /// )
    RParen,
}

type Token = Annot<TokenKind>;

impl Token {
    fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }
    fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }
    fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }
    fn asterisk(loc: Loc) -> Self {
        Self::new(TokenKind::Asterisk, loc)
    }
    fn slash(loc: Loc) -> Self {
        Self::new(TokenKind::Slash, loc)
    }
    fn lparen(loc: Loc) -> Self {
        Self::new(TokenKind::LParen, loc)
    }
    fn rparen(loc: Loc) -> Self {
        Self::new(TokenKind::RParen, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Loc(usize, usize);

impl Loc {
    fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Annot<T> {
    value: T,
    loc: Loc,
}

impl<T> Annot<T> {
    fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let input = input.as_bytes();
    let mut pos = 0;

    macro_rules! lex_a_token {
        ($lexer:expr) => {{
            let (tok, p) = $lexer?;
            tokens.push(tok);
            pos = p;
        }};
    }

    while pos < input.len() {
        match input[pos] {
            b'0'..=b'9' => lex_a_token!(lex_number(input, pos)),
            b'+' => lex_a_token!(lex_plus(input, pos)),
            b'-' => lex_a_token!(lex_minus(input, pos)),
            b'*' => lex_a_token!(lex_asterisk(input, pos)),
            b'/' => lex_a_token!(lex_slash(input, pos)),
            b'(' => lex_a_token!(lex_lparen(input, pos)),
            b')' => lex_a_token!(lex_rparlen(input, pos)),
            b' ' | b'\n' | b'\t' => {
                let ((), p) = skip_spaces(input, pos)?;
                pos = p;
            }
            b => return Err(LexError::invalid_char(b as char, Loc(pos, pos + 1))),
        }
    }
    Ok(tokens)
}

fn consume_byte(input: &[u8], pos: usize, b: u8) -> Result<(u8, usize), LexError> {
    if input.len() < pos {
        return Err(LexError::eof(Loc(pos, pos)));
    }

    if input[pos] != b {
        return Err(LexError::invalid_char(
            input[pos] as char,
            Loc(pos, pos + 1),
        ));
    }
    Ok((b, pos + 1))
}

fn lex_plus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    match consume_byte(input, start, b'+') {
        Ok((_, end)) => Ok((Token::plus(Loc(start, end)), end)),
        Err(err) => Err(err),
    }
}

fn lex_minus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'-').map(|(_, end)| (Token::minus(Loc(start, end)), end))
}

fn lex_asterisk(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'*').map(|(_, end)| (Token::asterisk(Loc(start, end)), end))
}

fn lex_slash(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'/').map(|(_, end)| (Token::slash(Loc(start, end)), end))
}

fn lex_lparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'(').map(|(_, end)| (Token::lparen(Loc(start, end)), end))
}

fn lex_rparlen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b')').map(|(_, end)| (Token::rparen(Loc(start, end)), end))
}

fn recognize_many(input: &[u8], mut pos: usize, mut f: impl FnMut(u8) -> bool) -> usize {
    while pos < input.len() && f(input[pos]) {
        pos += 1;
    }
    pos
}

fn lex_number(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    let start = pos;
    let end = recognize_many(input, pos, |b| b"1234567890".contains(&b));
    let n = from_utf8(&input[start..end]).unwrap().parse().unwrap();
    Ok((Token::number(n, Loc(start, end)), end))
}

fn skip_spaces(input: &[u8], pos: usize) -> Result<((), usize), LexError> {
    let pos = recognize_many(input, pos, |b| b" \n\t".contains(&b));
    Ok(((), pos))
}

use std::io;
use std::iter::Peekable;

fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}
fn main() {
    use std::io::{stdin, BufRead, BufReader};

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        prompt("> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    e.show_daiagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            println!("{:?}", ast);
        } else {
            break;
        }
    }
}

#[test]
fn test_lexer() {
    assert_eq!(
        lex("1 + 2 * 3 - -10"),
        Ok(vec![
            Token::number(1, Loc(0, 1)),
            Token::plus(Loc(2, 3)),
            Token::number(2, Loc(4, 5)),
            Token::asterisk(Loc(6, 7)),
            Token::number(3, Loc(8, 9)),
            Token::minus(Loc(10, 11)),
            Token::minus(Loc(12, 13)),
            Token::number(10, Loc(13, 15))
        ])
    )
}

#[test]
fn test_parser() {
    // 1 + 2 * 3 - -10
    // let ast = parse(vec![
    //     Token::number(1, Loc(0, 1)),
    //     Token::plus(Loc(2, 3)),
    //     Token::number(2, Loc(4, 5)),
    //     Token::asterisk(Loc(6, 7)),
    //     Token::number(3, Loc(8, 9)),
    //     Token::minus(Loc(10, 11)),
    //     Token::minus(Loc(12, 13)),
    //     Token::number(10, Loc(13, 15)),
    // ]);
    let ast = parse(lex("1 + 2 * 3 - -10").unwrap());

    assert_eq!(
        ast,
        // AST - AST
        Ok(Ast::binop(
            BinOp::sub(Loc(10, 11)),
            // 1 + AST
            Ast::binop(
                BinOp::add(Loc(2, 3)),
                Ast::num(1, Loc(0, 1)),
                // 2 * 3
                Ast::binop(
                    BinOp::new(BinOpKind::Mult, Loc(6, 7)),
                    Ast::num(2, Loc(4, 5)),
                    Ast::num(3, Loc(8, 9)),
                    Loc(4, 9),
                ),
                Loc(0, 9)
            ),
            // - 10
            Ast::uniop(
                UniOp::minus(Loc(12, 13)),
                Ast::num(10, Loc(13, 15)),
                Loc(12, 15)
            ),
            Loc(0, 15)
        ))
    )
}
