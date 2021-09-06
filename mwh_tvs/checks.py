from operator import eq, ge, le

def raise_error(module, intro, checks, err=ValueError, quoted=True, connector='must be', comment=''):
    if not None in checks:
        prompt = "Exception raised by '%s': " % module
        if intro != '':
            prompt += ("'%s' " if quoted else "%s ") % intro
        if connector != '':
            prompt += connector + " "
        prompt += checks[0]
        if len(checks) > 1:
            for chk in checks[1:-1]:
                prompt += ", " + chk
            prompt += " or " + checks[-1]
        prompt += "."
        if comment != '':
            prompt += " " + comment
        raise err(prompt)
        
def print_prompts(prompts, either=False):
    if len(prompts) == 1:
        prompt = prompts[0]
    elif len(prompts) == 2:
        prompt = 'either ' if either else ''
        prompt += prompts[0] + " or " + prompts[1]
    else:
        prompt = 'either ' if either else ''
        prompt += prompts[0]
        for msg in prompts[1:-1]:
            prompt += ", " + msg
        prompt += " or " + prompts[-1]
    return prompt

def special_print(x):
    try:
        return x.__name__
    except:
        if type(x) == str:
            return "'%s'" % (x,)
        else:
            return str(x)
        
def is_bool(x):
    if type(x) == bool:
        return None
    return "a boolean"
        
def is_integer(x, category=None):
    ans = type(x) == int
    if category == 'pos':
        if ans and x > 0:
            return None
        return "a strictly positive integer"
    elif category == 'neg':
        if ans and x < 0:
            return None
        return "a strictly negative integer"
    elif category == 'nonneg':
        if ans and x >= 0:
            return None
        return "a positive integer or zero"
    elif category == 'nonpos':
        if ans and x <= 0:
            return None
        return "a negative integer or zero"
    else:
        if ans:
            return None
        return "an integer"

def is_large_int(x, category=None):
    ans = type(x) == int or (type(x) == float and x.is_integer())
    if category == 'pos':
        if ans and x > 0:
            return None
        return "a strictly positive integer (scientific notation is allowed, e.g. 1e4)"
    elif category == 'nonneg':
        if ans and x >= 0:
            return None
        return "a positive integer or zero (scientific notation is allowed, e.g. 1e4)"
    else:
        if ans:
            return None
        return "an integer (scientific notation is allowed, e.g. 1e4)"
    
def is_real(x, category=None):
    ans = type(x) == float
    if category == 'pos':
        if ans and x > 0.0:
            return None
        return "a strictly positive real number"
    elif category == 'neg':
        if ans and x < 0.0:
            return None
        return "a strictly negative real number"
    elif category == 'nonneg':
        if ans and x >= 0.0:
            return None
        return "a positive real number or zero"
    elif category == 'nonpos':
        if ans and x <= 0.0:
            return None
        return "a negative real number or zero"
    else:
        if ans:
            return None
        return "a real number"

def is_probability(x):
    if is_real_inrange(x, (0.0, 1.0)):
        return "a probability between 0.0 and 1.0"
    return None

def is_real_inrange(x, rng, closedness=(True, True)):
    if type(x) == float:
        if x > rng[0] and x < rng[1]:
            return None
        elif closedness[0] and x == rng[0]:
                return None
        elif closedness[1] and x == rng[1]:
                return None
    left_bracket = '[' if closedness[0] else '<'
    right_bracket = ']' if closedness[1] else '>'
    return "between %s%s, %s%s" % ((left_bracket,) + rng + (right_bracket,))

def is_none(x):
    if x is None:
        return None
    return "None"

def is_inset(x, _set):
    if x in _set:
        return None
    enumeration = ''.join([special_print(item) + ', ' for item in _set])[:-2]
    return "one of the following values: [%s]" % enumeration

def is_itemizable(x, types=(list, tuple, range), is_size=None):
    #it is only valid for containers that have the len property
    #is_size is a 2-tuple (length, compare function), where as for compare function it can only take
    #[eq, ge, le] from the operator module
    if is_size is not None:
        length, cmp = is_size
    else:
        length, cmp = -1, lambda a,b: True
    if type(x) in types:
        if cmp(len(x), length):
            return None
    size_prompt = ''
    if is_size is not None:
        size_prompt = " of length %s" % length
        size_prompt += {'eq': "", 'ge': " or greater", 'le': " or less"}[cmp.__name__]
    type_names = [typ.__name__ for typ in types]
    return "a %s%s" % (print_prompts(type_names), size_prompt)

def is_each_item(x, is_item, same=False):
    #to be used only when knowing that x is an iterable container        
    if same:
        checks = [chk for chk in is_item]
        for x_i in x:
            checks = [chk for chk in checks if chk(x_i) is None]
            if not checks:
                break
        if checks:
            return None
    else:
        success = True
        for x_i in x:
            success = False
            for chk in is_item:
                if chk(x_i) is None:
                    success = True
                    break
            if not success:
                break
        if success:
            return None 
    item_prompts = []
    for chk in is_item:
        msg = chk(None)
        item_prompts.append(msg if msg is not None else chk(0))
    item_prompt = print_prompts(item_prompts, True)
    if same:
        item_prompt += " (same type for all items)"
    return item_prompt
