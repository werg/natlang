var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "H4sIAAAAAAAAA+S9e3PdxpUvmqrYshS9JYqkSD0oiZIASZbFhxSHkZU4tuPJ2E48ziSpmTlzunoDjb0R4iU0sEm6TrHurbr3C9w/74c9dWut1Q10Aw3sLcVz61SdfyTuXguNV6N79Xr8fl/87Gc/W7v/s5/95trPfrYmUhmUcVGJjE3qOKnijEWlEBfyQpS8ysutTBz9YjpNE8YneVndSJKUTUtezFiQZ5U4rg4OgsndJOEpZ2keioRNuBQHB0EpeCVYJTKZl2dlFR4cPH/+fDPI0zTP2N9lnh0cnBq/PP88Y8ExZ9WszI/W6W+eJHkA3YjjQBRVnGcrJIBLbBsv0vWFIYvTIrkoqzLOpizKy5RXd4MkLtSVJTkPRXlwAP+rK5O3HTcEDyJkWV6mt9QVFmLKCl5KOFo/macrR9Oi/l2chV+XeV38IBLBpTiTijRIi2v0RJJ8yuKsEmXGE7rKqahYmR/JS6pn0OBheBmFpZAzXgi2F97vnRhfD11+XImSJ1uDV57kJWdp+skkz0GrfTX6AcBFHIqT13Um42kmwq34+fPnb77OknyW8iw7OIA3wvgkZvM9trPLXhwcTLiMA3xTr1s1OOrggEspyorF2ZyXMc+qc3gvcE0X8K+0TljKq0t0i3kh8DWdkVWZCPXuQAUat40RweY8qWEkdZsu0xgQWcgCXgWzW3SLs4KXPJUHBxkT6SRkM8FDdnhn5EFWfPqBOC6iq/gi6ygSpXqL9uvYDS/hKMKHDeP98cInGyU5r95cpyud1rwMWUldW208eFvHpRi7gzk9xEKUaV0JeohSwOMu6sq+zv3wYiPOeCpu4i8aE1FeHsEZxXHBs/AB3gB+lXo4f/bZ62DGy2fYCi92DY+OJb7JeFrntcSRu2c80Tqr4qT5Mk4HJJ6/an33zSnp1uaxOGK74WVLh1eXGAtPMp7GAQu4rM7rURXw6io9w4mYxhmNgX9ecuziLae8eLZFY9ccaKGQVZmfGBe1F9LUJwOeiBu9lwTvhy4rm5xUQq70NMq8WnPe+n/857qzO3Y4X7cOMH6MzQmy4IHw/MeO7+e01+b5n7/np96Kbg3OPlGUXbXGOKwbOC3OyseOo057bZ5/hibv6zSxl4JJngqGo/wqformxHq99ySj6N7IsypFIXh1o/OYs4rHmXw0eFtxVrC8rlgcSs/fdH+wczZ9ywdkhyDbdnTfa9qf53FIHydjFax0E57wLBCMR5UoWZzBfPu6VYB3c43GoThSCxvbCdeNCQpO0/x4aM5coQhifAl2k+ffH30URS493zFvHRewFtRShDf/Hmd/5wcHUZ0FjJdTSbNjkct7oz3D07qCd0ODAKey4WU6mItgjf0lO4qzkH3Bk+R7Uco840lcnQwP0reH87Ehwqs8jYPeM90N746NqzoR1/qT5lVsMgfsSrdfWPpWzSWmtUA+CHIZnZVZUcZZFS07zXW/W8c0RytJVfJMFrkUGw47CV5Xvbd7Z5rUglUnhVC2nf3bmC/1AlSmEi/+TJnXWRjddBiZKdp2DxYYjvCeftd+DnMRVHlJIx/vlu6OMS7BkGFHcTVjMv5RdFSGRw+vquw5LP8HB3+a/F0E1e+4FK+p4Ys8LegEp63M8z+K6V2cmYpKZPMPeJXHHt72kZhMi5ql/LC5/gmMyinYh0xkVXmynnIWlixKeMDAIgBLNODBTAxPPHCJ+GEczj3/bKTOvuaY9ODr26Q1nweHYB9N0KiB8TjLZXWn/0WKTNalYEFeZ9VHCb0ust/AxL5mzZEiCz1/RfWBg4kFeVrwUqzYS3dZ8hPPH7ZP01zAKkEDR8bTNI/DjaFVz/Ov273jq/4gyafRhcbQZLvhmeiojCvxqTl5zricsYpPEvG6bdFjJsxZKUDjdVXWQpk7+sGp1ydF9YGMs+iBerVFXIgkzmBV7bR4/m/Gvs1QVDyGWVgci7L3eU5F5fmXwPx682YryDNZPfK/fE+L5rSV4Tql30EpZJ1UBwe4TQLbEWVVfiiy+EdRsiLhab6rXpHxTe+Eyp7nsnpMH0az5/kKhvTBwand4Pn/5tJ7Yn3DrYZ+HyItEh4IfAVMJvkRK3g16yhelNUkZrCL2nmViHsuE9yaQF4tstRZXjJeqjkFRy6cxh7QIi2qE8+/1h/lWyPLAVrAv11i8oqzuBqcuvT3Rpta/N/zVxym+Y7aX6Fxw/bDdddEAB/1WlqlsLdIClHCsjSlLfH0Rn8BiqLsgbrDYMYrlsppa0/qFs/fbncTpZiKY9jtBTO6ERjUDG7E3gIUtZzhxd23mnFnW+XGLvuJ5383MpoYqzN4fjFP4h9FqD0GecmCvDjBhVW9WhDAdVxQz6UsY1F+WPFsFl2cTuuIRTBVH4rRdyoK6fkPTYM6rkTasbGhyfPP4+MUWZCHor+E4f4Q91qDSxiqPHm2FSWg8mjhlhMU3gytZbC7daxl5uc0Ef/TuHNRyDiBv/EJPDMfiqx4WbkEMNAczcpf4exKvK0FWLYOWTDLY7cExngVg+fHJeXuq8hy5zXz7EQ3P24/ABbk03mCL7bb5vnXKslkPUHLXG3qN6xVETY16i/Y+pqug+bv1e7nSXbu40UTCsszwfLo2rfffse++Wv7nXg+OFSCtOhuU2PJ5BFf7beWIig3pSjnomQVl4cHB6fGL8/vWb/74Q3Tn4NuhlmcKScE7pHZJOby9ojT7H98dgU/Nhy0czQibR/GTmgv9/ApeL69WctxBHv+R2HOiroK5susvHglTJQl+DTfTV+vKTSTZHWSoF24bb10XA/g1cOtKStXev6DkReqR/9ZBs+0ivdu1lkM8wULigpsxKlkypi6bZ2LHgCcjPaC55QlldRnGZOzqaiC/pQTZzThaKNjyGpm7Kjk7UxZxalgehcTFXu7rMpZVOy8Gts/kfXx0PzOSpFx20WETZ5/1Xq3YZ0WY+tJc3X3i1Kwo6lMDg6+L0VR5oGQEt5V0fzaM61yOYPpkiXx5ODAbFfmgrbjlAetTtHXpX9NcCW52azFvBKpyKqDA3EsgsYtF8UJuTF3jVt6tGWPXe376qzzbo/cES6f8N5A54OqjNO76iNVNjWeD4ZSFtJgbYziOdsNx0ZfystD8M3taB2cGsDTUiRoapy6BZ6/qi5BJnkFrwinQM//9h8ygMkUZWTneD4NafTBuwwZOACG7NggpAVksx0nXx0X5bdw/oODDN1MT5fZdLFYHnHP17amqDgLxRyWJryIIokrhiPiZncRiGEXDFuDsddAV+n5dy1vDYUJCgEW21vcGY+FIv7HKmNFNSvBgQhe4mNWZ0keHK50m6HxMowidjSLK4Few3VrDTB+VMu8T3QDLD+rturNpGqPha2Rx4BLzO0BBQ6TnHbBTmDCYPB5gMWr5tdob6O3jWu+tftub9XhES+nsEcocvlk5A28fu1pKdyFb9rIes097Td6/rm59u3cdPsNpedfVdemTxrLS7iI4sMEq+Gy2qFlyvwfNWFhF/+xehBhLAt8VKGQQbuptZo933Qk0gcw5wl8qb66sJLHUoRtQK7fcqNzC+AnDCoyMWQeVSzlxziVnVPWQFLTX2Dc3+zvTODTLPnRo56rtBRpPhcd5+gLVBv0fsy52iTIGacD7o+FlFiacM+/3IRa8roq6mqlDW9hYxRJUZ2l2A0Pe8bUXqjjgUnNRBndtGY59R7wXu9ZEnz8nYlnjZ5JxadTERovfQN3eq2jOc8P64Lm2X26wcM5eZ9YKOf7RkQ3TwtwCKgYLf3duHsbNXmSBbMyz9BEu+OY/eI0rdH94vkX9JOAr5KiKkHC00JHfbQj4rH5kaCfovPhYJvnowH8r39kf/ju+2/VrvhcE5l9bHvZeMgynAvgd1RCKAHepChpiMEe8M/B7hE9TFzVbF9FK1vgq7AV22AcmBIqUMTCOJU3rFcaijl+xWNrmQxmIuXPu5vlVKKTZmDj7HApw7r2QZEfRZuDS18ptQWOK77yGji8A/O4utf8xfKigjnObvD8C7Iqq/yY7ezv7f7qoTVGVJy+6v6+bO0y4tDlPg5wP387FKKQQhyy+T6TeQlGRJ4oV5e0/TV4jkfmSKrwLVaChXEJ5jQvioRcBtYgpEePFpUoK3gZmGoA3v7xCHvfrfT+fgD0CoLKU/2V4/ZEO/VOHa2ev65292V+pO2mOpM8EjtqlS+FLPIMj6DLPnULPP+372/byQDmDsNXuxfe6AwwHsAyMTb+MzHllbihQj64gDZaZxmLYBDkq8b2SbKozFNWV9GnZ3AIhtfU+CoFfbs8SW50bpfs6GWDo13Payu6SfPPEZ/T7FMEKc078rY5M4UCr7adlrZhnM14lsE6ywvyDMqKZyEEUpXkIo03tcc5G0sm3tY8WVcuWxmjcSGyStLnSpMNfepoZ1NKDr2P4sR4Mfuh/cng492wrJYyTwtwIheFyJpgWOsbLEUW6r1JVrXepb3dRGy5fdERxApZlofasmvWpHY5Kk7Y4ZotpX0eL8XYUj2npfpcs9Ju9b0u+KXCZIJf2zX0AspCBDFPaFG6YUw/EVrRR/xQjOWVlCJ6as0gQsrYTHgyWz1/l9dVrg0j8+8gTxKYmIpS4EAN6YKk5/+CoSkfxYm4SLH2UmDi1ObvaIL8oU4gSGH88vz+7KMuZ0o7aufsY6t8adpbZIHQxKINLlmprswVsiPc6Hs58FsNiurjng2jHTE8OzmaiVKge08cVyUPqqU2nNRNb1ZqLFOWCin5VOygsRSnfCqYciGAZXdwcOoWeL4arzVuNpoQOv3E8Hk3tInRPnnEwViASLAyEGSKi4uyB5Mal6Cfi+Pie+NrfOJ6cX+X+SKLxFJ86lz+piIDkwHdqfh9UxodzbXTkqcpLy8wtaWEIffREWVz/Dyvq4FdS+n55414xG1XcA2/PLAhnaG3qahuunb+QSJ4eSXKS8GDGQNjHq58y7FiN99jxONkqzdXdRpu41puzX9TUZFX3fMvot+pYFGQ5BI3lCINipMIv4dlhqE4ptg/w/BeWeNr9Joj6TU2ss5wvVRJPclAxOaMhDU1WtN7dZwUDg6KXKKd66l2SOOAGaadSOlTelsLCc9M/rzg4c+TfPqyO7XKqhQ8VSGULMrBMOg0eT4Z8VVesEMaumE8x5Fjv054y+QvudWzevXfpZjuqVAyU0mEsrVtBiStfxFdlKngxh67aWp9x9rYoP9Nd4ryTcJqRE/Zd615p/1Gz19zbk09/5Y9j9s3cA7m2Rk4fAfCgwWPy9dj4cFG4c7QrEUrtAoGzPOATw4O9B7rutUKBkQ+vd2frmAsZTRBn6MIGewDmizWKC6lbU2I42Ld9SnDoHF+y7CuVTdECjGqL9Ab1AT5tNlhrhVVjvbc2NLLw/BjfCX0SKK9XaanZ1eza1FsA66Di6Kt0n+J7UgZeIm2wlfvvKpGZZ5VHemDZkJvHE6dFs+/25kyTq3fnn+nOxPYvy81sRteliyjty7flivNoMB9Nr3266YjWu2xNhxbuVKg+bXZ/yAhFoAzylbrHGe4z9tT9/5FQvkci1OdbW/7c8tvBIYTo0gDuiZfe03sAZQ9/zZ9ZeB5YMGszg7pXtVje0bn1fOLGuP62XWa7wxd5JGIp7PKG7KBm5Rfdsji8Fg+GVJsJqFG9X7XMwMuCgl/QAgAc/WuWi8W/IoqpFSJYoNeLr6ksk3JB4vddk2VYopd4CTEw7AUUtphTjSAJkJWjqWAZgQ49XmV9cmzqdCTNwUHqpwVsQiEHivq00Z3WD2hn1dtGQ/DP9tffseSamSLLClL8QJ8diWOFpast252KaoAUjUTdAOdZfAcqmjvMfjT1bpFHiLcs3faPN8fzYxMRcpmJ5MyDjEcMRa6SGd8cDuFn8SdfkA4zkLYwLOoTpKt7pihj6R132/GWdW3luKseuP5e5ggmMop5oclAk3LUsB8MyDRezT5tqwuoE8/g+QAEY55snl24vmblkkL3h+8I1GJUt6whwJajpDGA28yY2lalPnf0Vy963Yw05Pk8nCK08syZh5KJD9mYZ6yga2HEVnuyIr3Ow84w3C0vvMJzzOG7iUZV1F/JWxqWOLBldBWOa9cLH/P42y9a53qjJ9NK3qodhzq3fwfy9w7FJokcci0z3OpB9Y9aCD8tI4zfZMVR0YhfDZ/RgGvwzhnqUjsSaSVLZhEbMXhDMmAF1VdimeL45NZTp+35/+Luf984niRIFlwfbbiEzspnFyt+wcH1ocyC8Dw9JdYt97it9RNXJ1xiVGTjW5wASZ1Wju3O8EjGs8x5ruW2ovSjTBRhPdWL3TG4lBkVRzFolxTwoAXkmEQDudheWOq3NF6wsLwQpu4oqJfD6xFDDadjWEr66LIy0qEj/rBG60Cho1OU1zvr4cnONkOBr/BRD8jjot0J/pQHBe70RedaH0hgjrhVTwXLCx5VPXj+V0Nu3zESsLt5lPbE+up+dPzL+i9MqyI219//ZfffycqfnDw9Td/fd3+agP/h/PVthAtq9oU2DVItKpghFJUj81FCS97/e2RyPbFccFKgfVTEGb/UZT5j+/lubW/le7kOfit2IrLnto+6ic59W1jD6hn26wx6a+jdIqpFWC3wSbucZJn0y38p7eEN6I3mB8B3yQZXyGvuMpyKngJGZcUcokzvEYVJIYXKtKJCMM4m0oWV7N7ZkierEioV4JdDu7/2tChEZk9bf72fJ1c11YMNXVCa3E6xVUfrHVYkFgRJ0l+ZNuzcOVkxb6teVaBGkSW3n7KXrD940/J3ozT3SBPbHsVE6Bh6/bQ8am3vnQt2DOMGnpmFBpWPgyHxPPHDBxI/pFmtulWW8fR8ey9Go9YhaKoZlSPZf3or/nNlzhcW2Gr3DayIWUNvkJ4vsEsTsJSZBR7iT6C1I1JFn2xzFItRUnpvP0vAnaEv37P8Awce4fCL2mxp4MvFBjGmLA43oX1lpa1UCQVZ5maaalKyyW521m26L2yUuQlBs42+wtQkz6mc7txnFZVGU/qSqx1R7v640p3qFON1FwELMwr2NhcbLIpiqSWZFCngmeX2434TsiK2fPuri2WmMbvbPZ8bIcgTI2lczw4PDhQ474qeVzJ1xNwDLO3O+wFPemjvDzEoXImipJazryRMY6vMKvTCTysfuaoYe0H+4PrV7NW9UUvhrLbhpLbVtT2QL1ImkbvDFRHwewSZ9U2ihuTBkyXJE7jypqUzfTiacmzuBL7GK+DmZeGFj5Iw7Uh4+DzZPp9nsTBybOtZdJNl4xI48eu9rG4VmOR5lnt5P+IMRlnYQT/z5I4/YixIJdh1Pd09Z1lHU+XrbA9Et9pwjqYyDwrf9FEDz/d6uwl4F70+ytFFB+vWjlsTVLXSrP2gakciojXSXXDDMFi8sAkruSDoWemhjqYbk+XtYdzWT1bUjeNj4V81ne+2mVwEPxmUEN2XMhVKwEObMWEnwgorV3sOSrz6vmQnrv9cdcjCNu2Jiig/8ZqL9BTeURpob0cELuVQV7q6K5T4XB+d9hIwOj2Zt9EaDaUZ2N5hJmMz8AnoZx6WLsIyU6YcjOFacpOpl2g3K2Fh29uxrMwocRttUJdaJPZxcl9M63YqP+DHXE+F2XCi2ErZF8lSIEHYzc8q2tG7/QD1hSspq3A4PYA9g7X4I9UhDFnOtdXV9sYEZNuy3/X4RJjexDkWRRPnSFIWSx0nFmKe90p3orzuCSev2eYQIOoEJ0tSqd0seRSPFsiq6coYd6O5+L2wIxPHg5zLofNyF4FVlBxKLIAF+z5Togl6w8MvSIPDkWFioJnotKKzXYpL0R2kcEsKLIwjKpoF4EpvsTgmQKm+DdXhmansq7RWFRZZykOlNk2KZHDBQONym/aLjBjXydidTL3257wK6PdHgiuywoy7ZWnhr62a7QBBG+zLra5ZTiQaMMOoS1V8bhlp3t/by4yn4eh5/+8yI+OlzFDodg/jxi6oZdyMZkHDLiXVpsFiTxJKolny1iS8ul0Ihmb5jAbgXFa8Km41Q5VI5lIFchfh9ryVCT4yVZlDiH7B4OeI5XkWPDsph2gmJZChCeYiXjNlsCWcXtkHWj+vNlVwoVaJIl0VCUVTf1m33iWeTIXkDapvEXoHtRzKmzn1BVqXy2sQL5rCTntN7auIqOmMs6mvtPLBBMCL08sV9OGUxPd0U+UCFPhzITo006L569r6z/JJzwxbDjLLVXhubHI5Ka9CIWiWYa2p8ok7FWXnmqJ5z92LE+pKMEc06sUDl3aAicQjcqgUYUFaH1KdrEm47KRUwafprHFgFjf0d/6bi/L38UU3gbmHC6t6/n9xFlY6DZ6rSoMVcjbrgUSLKIacn48S0oLa1Ada2woSkuCRfeSgqJQpYxXdfpJE2vopsyCFXzX2mJ0/W+er7NxMlpkdVjI0domMaQizWGvJsVbVqbWiobBDwgF2KktTfNTx/LX7K404ERI4EwP+rowp5+wiWgqdNbssNrBASVcfvrIDKZor0wuqV6zdQ097MdcaEXDManxjj5rI7+0tccwbcc6w6iw4KWuEWtqfX87tJVaNkPms6X2Ym05nr0ZW6WK/sah+Qzj2Ga5NOyrMswJK+jS8QuES980E0hUfr9+k7d4UZhzi/XzDGNHPIa9XFSJJMmvgQ3xeciLqkG3+ugoKKs8nTgTF5SbajhxwVB4aIbRFa4I49Vr9XiLEveYH8kTCRbkCm3oChlq50uZiCvNLk8ZqedpwcdAzc9lnH2iOpvWcYilFRWMnjZ1wW73/CfD9XHabanTfW85VSlCJM6Qy+pRY94Zq3ZTPtTsra4bAWDMHo3i4EYvawAW0ofd1ibDsG3y/L90mxwBJ5QuE3BqFNc65sdU0Lp9BdsncZWS2x7tDXcCM0vEzdaJxmDKgy00zUub/XSjJmZ/z046wkdta6i1ngwUFQGDS7xuWyOYpn/T3DFb6Yw26hFMCmjSeH7XBIHA0+FcdQRpjY21AD+aHW0zNWirhBKW80KqEDqnbwtR98xUjsdDpkCnYcVY7mH654Ax4N4ofgPu6ksa/AEGKNepV0ntQnuTAhKI7fWxWfpTQDcRZRTdDOMoosSjg4PT9oftnVZJOY1xJiKwVXSOMq6HUOqDhRHSH/HkqSCxShIfQ1QqhDjcbFdNKTD/i8HcUeRxVjmWKXNHh8nwN125fBjQ0ylmOj3A+HWG4Yb4EmMSgIooOBhR2qcU1TWEmoHPY+eVKiU6D3s4QNQI0sLAE/ySJsUhWAbY2zlgGe6b82opIOwlOhlP601sjAKX4APGxfQjtRXaxPSSNhWs/eX5vt5BgWOmieL1G6GERDeSDQAfDS56Wz0BmKW0UiAi36OOQhQL3H2Axdf87fnXldrbWhA4VSTKh93lWcgAsAGstHx5ueQZZD9oe2RVT8G4pE7KnIeAWXN/ZEP4l4yDh/vR2O6urcb+/h8GzeuY5xsmEks7K8MEvNqrYIuzam/3gY3ZM8FiQ1VcoJtv9ussVJzt+XB4Q5V+iaAuoXQE8wtumHMsFjaCsX3Tml9p8FkbsEUpaWVerZqTrl6gxeEnAy7CU2d7vx6xuWkYPreHPYpSvNV7LNheNbi3dRb8i26Hh/nE5equ8sMFa66t2N1sUmIuBSgP57RgQI34jSac09TDRnu7KsUyzm4o/IWdV4S/sLcL22Pad4Xgu8RZ/YIZ6nEm3UuddL/i2jSt9hoRGcy3moOipjAvuOvNxD1ABHQX2FACuBOSCBbnpUNM++ybTojppjIlsepAzWRUhbTtkPSabjRzjC4Lg6H0zOEDhWHKMvCttAVI6LxZ6Wz98GMbW9Wmkyx6bu0FF4al3CCin332ycAG8NTZ7vkfuwpAuuqq2fOfL3CYqmIz7TJ11WDgxufg4DCGqnOrlAxrMJo7uaqQ4RAZ8Bgmu/+2dC5cjKCYgPTUnXkNJA1bdI6xRM7KKt77iLFEVNHuIztPBcs+cH8Xh2COgv++jENV+pIWa7B+/xkhRb7LwzrR/tnV1mOrEtpxlA84V5sw3LBztVF53XZBCwpNKI6tMGzZOp7Vr4f2sf3yAURRYUmU7qr4Fy8xIRMKhOQza8OX12Dv5SVgK83gWmw86TN1Bkg0F1tLpZYzyOlJeDat4ZTyJJ3kCU4BKtonD+NC7RO1v1M5OQ0/MDjqkzg7vNUJC1o/b3aEWNiHW6lbtslxgsZIldPvp509YYpD4tTROrTVbBw3YEYVZR7WgQg/kJPy8Lp6vgyfOtmiZ3EF/XNVPh6xU77Np3HAk8/hI7pEhjBA18H09M1PkFGh7KvwDz/N1/bE80fzPEZMJSwvkjNYro/+pvauqm4KNu/OTXAlyuU2wUrxt4NjPuFzzuoZhDQSwHdpsnfMn57/zEgFVUsBIv+ACz6Kk0qUE54dwp6tjI8BqHJ3PK0bczkR28TzHy6BSOP5T1pDLuXphPdyVNrWi9DY7IL1FlugEaRGZ5xN7zlDAXqlhgll3dZASERyFWh7UJhOf8IBlr5pQuqsPWcQfNMyK0044gZ6BwxLiCU0Yt3g+Q86pidA9rQPkFJBu+aprYMhYLXp6Wra4BgY05AcMTGWCLrPweBdoqxjjrUaozXQ86WKPpwdqUxLnJX4VGctNb4Q8Laj1agsayVXxZ+sFOjgDHgW4nKoYXagT8j9rjDRb8OZAYhVZNdMgx/ckg2ojLkHIMHWiOGOcYFLh3Eas8M9RvFTB0JmXGx1QjSEN2Pk/j7uKAxlyG0baQJQ+QJXy5jG7MFsyUdvrqM1j65SdILDTdnpwvXert0Q7+1etRrA4jAQ11I57ALav6hdQPgZXlFAOZUoixwst8tNORYZ0Qo6v05lna52IjWqMGvd4TzCDLa7g7g0hFlnF3tSZmRbJ2juJODx3Ou1Ai6xuYfoVwZBhBtKxuJcIQZD+kdeVpemIk35PlP+jg+jtGL1XY2W0clqJT4Gz//fHOJyxcIYVvGI680e6FCAvxc2Tx8PprIWZT6RjJZg/ey/H9B+b5DW2509Fc65zezzaGRzpfIJYSJZqAZ330EplBWamAcHEhL08Gop03jD1CJPKJrFdSJ8Y7QNp6dAZrt7G3fvMxvyEga/5/9Hi7yAc3BVxgDV60zCMQAcxpNwlOK/O7eHzq6NGtbxrpXiZ66tpKSlqYuB1hV7vq7KVtt1SUuJSlpYtYWwCwDhDdwoGk8a925GTR76A2FSDSEw9guGm7koyY/OEIrGRVUEBbuXKloDPI1jPonnO+BJ4OFfvv3qdzu7nw65cP+lFrXDhduPrbXlfgOxNVvh9+8cu8Tu3tZ5JRDBJAHHgfz1gm7svQvWdWMmd5JX8oWywuAhQRjzCU0YW94T31PARXCXT3z9/8uhk43iND010dlUNgQuo7hFsPPcb6FuFNEFv454IsWzrarEV/B1s//EIgYDFs5RVo8pYTAxva274nXtSlUWlawntFlTO9MfleFaR1HKM1dyKj5SXYFhF7ydGr88/77zNQSHRUVe30kdbTtVMGEXZlJ6VQ9tJTU5dVGcrkkxhcRzsicg2io/e1fUxtP2h+dvqM0u2VXViZGItTFQM13k8vnSOzDU+ZWx3yIYCkxLkVXUoLM4hZ7/sSG0TnQ0i2WB6fjNsuT5H6WTssqPgs02vvntHr0GCTFm8HJv9pKHW/SmxwZ6U+Mm0yleRgj1DPEsfAg7w+hMkk93ishgKqFHFkNAELyStRTDtb6lJLiJ4XQvxQXG4nAY7NSuLYYN5kMzFAu5Ys2eq21q/O+qCQA6wdULZULiuOoJp4bwpinUoV5M8tpod6jdTYQ7Xw2mq31X8jZFYXHbq3kqShMve9s6CP5WCD24W1YpQDaKjyjb7Tgo4dPUbDoYAYcPVh4c6B+l4IdhfgSfQ1Mp25brqoT0xRtTeE61bC55QBMD483+yVJSmMUM45a33AriLSC4bA4Kg2L4QB6G3a0wCfsPorsVJj1H4+Jd+Cxw78LbQJENIdateUc3SAvH5EASa9mMwC3d530icHVK/fuF2tACe6IKWGG0oZvkaH8RsPKvJrGswOW613wvMKc/smvwa0yJa6Hl6bfn3+/sfxHUgSYRcACFLM8eaJxWgJBPwRnAqxmsQRSJB9eaXOvqqKxDOy3SzGPotlzvpkUKcegu3YUvwMio1EqOq6KQpIzn4nL8dvdYknUCS+B12Jh37GvI6W/2HHaRYrPHaCLoF1TIGjPk7zXABHqZsxs8/2kLXYAADz1Ag6bVLBWGAiFZqSSNqozPanjxGw0grJqSa1nl6VnVOt0wEizBXEU4O9K52uzuawhfs7ygvnDUUzU2AXgphL8a7UyF95fLe+4NPW7AKbvhpituiFuzf32HtM6lVV/3Ndv1SYccsQAByW3MBs/fdmBN4aRq+hTWej4Fmpbsqk54UhR8NI/1/B5yUaX5lO72JYQmGE8zMLOume4KQsrS2CbTlB9/ANGRyxANZFiZEoDZfBYHeFAdu2AH4+qxI6h52mvz/Lv97aqF53at2T7gtR2KE3leO4Wivd1vdFVIGQMQVB4K5+awKuMFm8M3mCLUC6bCTTZ0hgoBYiaCQ8zqUctwGFWLELdVjAuu4V5nF69uGqM+UEvPp79dWCC/QOFeR97EbJsw41NbQzuzHa3LFrXYu58rlp9AikqD3YRCb1k23BtseOBPXKJOsrBq/bUjlBvmQU2kB6cjUs//106U9ydxK6yZnRq8gKtuh4F+EXyWsyAvIeB02F632er5N526GaYBOKp8pgKhEUKh/kKLQ4rqxajy7hGfg3ItCQJwvhPew700rhQafhf+Nwrw98wcYBOlUSd19do8/yxjEHjk4YdYw3mOsbf4hZdXDIcIRqY3TEcHrsiVKNM445W4T5iioQhKLFRu13hI+8OUkA+xJPTm377+/i9/yACiNhB/wI3p33hcfZ6dXAfviJZobuSjYDDtGN9AuAAvTSnstR1IAh2yiE2c/Ji/NiPXQ+5IIOQLGqKXxhv5posyP8xAiD7xzqkfD3lFShMZa1rsDemNELV9PnyMuVcqRMaTKhay9dTrwJMF6TpKmOe6RSUeTAS3/CpXlOUKqUJooixy2KCNq0+Pw85y2DTiM1S2fNPAC6GSco0Y8hGYlIeJWEPui0RwLHdpOTDuktcaq9awJsX67fnbDvyuXtMFZbhirOpiizYtuEZChq9vzpNL9GsWT2eQdH1N/QzRz41nVhqTuMK7/rmclT+Xs2RNzvjuy1e0jWPAOUwl4xdVe12Af7yb+XAoREFoja+MMml96QHkNUVxxpPGeQNwKiznMe6aK8//3jjMns1bwYLZ3Fa8ZXnEKIewilNK4PcsoUrvV4mAxq/Wd9ZDMYf6a7xY+XGjAsM/FR0fnNXs+bfc2mRe3W6LGtBcoagqPXHP3xv0yp4OSDz/HILdvHnzqLllR80CvpnmFrvZKaSI5lIBhX/gaNLuBXd2SsLrLJhR7STOr/CcPkSH1MwE33lmxtCaNrMoqYGIaKQ4EVufdO9nnFWP/Htq6KlHqa/ccB6uN2FPQJ2L3+6zjJiVbth8Nmqq+bitrXVVaSAIBUqO+Nzzl0qZGcCyP4wLY9rw/DvWE8GP8eBAe0M9f7XrBCVLrF8ZAoGwLlIPrFGXqTIDY+HIyGoVYeEjuQLOp2IPUQcoVaMt3sBltMmhudE4NkMmT7IK6tfi8nbr7uzWepT8qFPsQc8ed0+XzaLVQmSIrQL+PZnXZSC6nV1POdNYn3qTdg3aUqsJ7oY2+arhA3imH9tLIDkqw4z3s59ZmPFfjmRJnw7KPP+TYT/rEZDVEUHVLI4qyu56vrw+fKLDblyVJVY+WSL7R9FRDacUGaqh5OhwcyULGS27DYKEVkyi1FKD36GAOTaKshWjkofpaIdd+nNq/Gq8Zv3Sn/hHcaP19yrrGK6/k22Ewxj3z53KZPhsOrrBjMcZHrHmEvAwvGq2Q/m7rjACLBGBdev5kXzsTl3qeYg/Xi7FKcjTSZzZjLUmlRJsUrHS9nZPA1L9tNSiYlKsabTjWRZvRPDMwjoJEhE5bkun+ae8aNOxUl54/iPL392WDLQOTnwtd91qDW/DriVXoYc4PLZdvBk7REiMTBxh/cLyh3j+r13q7bo8Im2w/R2ubrCmNpxSjF88cvmiS37UgV9drIYe6yU82wPd9X3qoRyHhJ0vVb+B6WydIIH2aeMDuNGV4Y3c6rYS2Q5l3Cm/fJwrwxZ8a+B+BwNXNul/XYqBOZu+5eSW/9hOVxMJ7lrfvFEZLoYVEmeVno60NpjkkJ+DO4crZlYafEIrZgMuaSU/supX2r/v9VPa9BGKeOfWQIULkRqYiW8wgVkNsCY+cqXG9ds2MJ5g4I20MYUNI9TQWhooWkGRzYMlbd5RvT/QBsVaJ/SQiGxazTzfc3r9W4d/kxTVRUdTSAoI+/drJYOsJvQcwzu1WeYGRavqWIhhtHhrsoetppLvdKRlEmc21kMTBSFig/jtLtFEA4Y1yyDAOMnr8pkzR5BLI0dwC5PlwYi82vP42ImB0d7uStOgi2BOCoBlsBuJYe5Ky11N7ZdbWgbEW21JRoA0ok0NBKKRvNi0wL1V6RTL8HY15WtbcmQX8Bj0b3ccWYCUsonrz7pDTGUyA7iKGR0tXf2SuejoF/DHnXVMIMB9tO3KR6tC4azaxEDarkoLePVYLtXPS1QYr5LlxW1HIIGw0MFHqXz+IaxVFIbhE7muahoJyFlZ81kmyus4OUBYs0FNkOvYZuK3lAIt0LMhYsgcn1wJE9A+hvUM4w8XASEoh1gb7C+uNLvP4oRByv+Bwynfr3CC9U5Dfak/W9iK8UOJ0qvvwDeKe0+qWd4Svi4ECdSsz2MU2bSfI8rkO05nfOOrv+UUY0JtWyelNr+dZFLbKd9mrXU57A/FSSs0neVNdtNnDqFRYeVytDfiNnPRiehBGHl1Vi3UmsDm/15fq5n1qbTPAhuBh28UdCuGMnz8kNol/9gJAjCoKn/PSMCbFresrRzTS1HjptptfcRNOb4avEOSjQGMr2hv97abRCRIkM/XhD5RVLMZ+IgcRKVYN7sbPnIjzEO8JYfEYQQ98fx7qCayoEWDsRoACAYagupYy4LqGGIuTbnXk2XQUvSCDEhjgiZIcVx0isZ+eK9Of/gK6lETVdv1k3T51Q9fGWtyp8urZiYUOUR0MbNOoV5XNc+ygNQIyI/I51i2f00JooTL2S5+Wv/sBpyxsQPh0gjhjFcqV1wEKe+EDQa66t/l4q5emJAxKrEhOcHYrXDCxzw3DwCwrRxtNgZ8K6CjrgL+Re4Yx5yOMU81PYKT8uEQTd1p+8Pzn1paGU9OfhQ0JYCi+dvzP2KshjzcS6xNz2VBsdZlra7KE3ACQeZulDAFUJqAdX6OsQj3DMf7ZjSLofOCAurEWaGAyhAkiE0EzC2hrC6YB72kfN4fBKL/KMScP6E5CdfubPf8s0eQBJjV6ZmjQCYie6xGijXIwTx73QrccHt0eiBAGqwItVWGCBvAtbKAsEGrDHUBc+KCLrTKYDTPSEp3R/OUwiAtoREOHKIl7AcEU/0pGNE5OyCY8sJxDPqu8c5GjjFqcJug4et2Z6trGDWJmy1xHtxur7oH25KBECRjzZGjIcgti2EqL6v9LoHSeBzRct3BUmknT6q1M+VF0cDIvXkznA4+Wt2xbV5rnFVlDhfcuV62NPHeotpXmBenwgA+Txx55lVZi2dbkG0OwotNWTjA0VzWFGVVnuRHoryufxOjFG5kbliUl5qYdkW3YiRPsZtdN/DUIT0sETwy21J+KGBevNQEO9F5cLmlCMSd1R2DMrDJvmRAAc5h5cMybMQhUkV8qn2lbQ9z7RpZ7eTDU9xGnoFd4d7uX3uBUNvO69c9DNh5tuKGhV9jwdcQqPyxZpsB0/KmwhbLUtaJ0ay0kqmo1Ns2mWknYtVVcL7z6gMZR9UFFVrFONx2J7LaQBAzvKFZDvbwjY4SzSQ/WK3uKOpCZBFb8eWCaC5CHHasZAPRx6TfkXwu7DIFoh6mD7R7ECSmK4CoaxZoEMZc/W4UNS8cEde8aAO4qhAfxnFWF5CsgOF56G2nGz3NNZ6mW9B22o/EznZzKo9xF13oRH18Q2+WCLRCsAv8yL1Gz3/kPFxhereZbvpR0ZZKZ/X0Gz3/96pyoztwsNGwSgcHDiq+uVImyJkC7iucHS6VWLMuGA3zG1BvgCheCmgBv/8+3wqktSYiezYKjtcJu65aMd3q7S57AdlE3eYdar5hNSNIdr/1FfvG0fpyoHXH2erqd9/Zw76zh31nD3vOHnYHWh0PIptHxX6/OT1WzddsExk+1xWr4E2FxldaWAnMfcEt/Rgm1p8hXun5Y3AUP4gEk8441OaP6H2H6N3bTZ44eFJzrLWe1FNdW5XFRSGqu4NKISYVnlOcT1ml/pqJ41+oZPIZLy9rt85WKBJRid+NInpBXTNm9PdNRd3Pf/znb963C/UJLgWmscg4wbf1+38gXwGmalWcXPzkvHKnY9qe/3+/F1GLyiYZte1GWF54F/rm/3qvjt77/O0AMq7jzW/fEw+lMTA+f88O2sv58j17UKscrWu/es9OePXQSpSpIbXgtNvk+WtWE7q40SX32E48yo9k93Bo66bjtNQ2CrTykhIr7/lFi+zPkasDs+rjXrMDsFXqescemB90cUMl9lQ62oyW2T073Uf32jRo2tk2Ym3+1Cck9yP0rcDTITb6aqxiclj2oCeiSKFiQSQWm56OqwbT82+lcRYDlFe615Z66ADi/VRIeARSURiWKU2CENFTf59LOTtCSgRIZ4JcprBspxiMn90eBqmVO6+MLKgGpVZWYZx3BFTfCYI21SkEUiUqqsAOjYpQ7AcQa/XJXtnY/QqtEUD7FTIo4xF43nSVoyo22+0VklY5QNuW+XGcIk9bU1cKEtgh9jEh6zirPn06nneFThPt9oOa050X23YOVD1N2/yo5gX4tpIsDCVATYQhMYkh5OOPpFuBFuxiSfN+H5qy22IAIDUGcbfJ83/TLZfFpKeswWM6HZV7/oJUJznLywq2g/4ixMym9ckizRZeZ6WBZEJPAO1ibzscLW1Vx4PWbTdIxLfiqpzdtHOgINkJNSY8kx1cZYxt3LbbmsgrxTpV9lUhykjPQ00ukTCevdXSZgy1OgptAIjeIasX/m+KjsmndIygt47EpF5e1l6au/S2u3q7i8p7tRsr4T+e4FeEcLdWspRCHmrSqkohAbzKgrlS3F4ML6tbLOxZmkn6EkubHVdmpYGZiV640i2b7QUz0O4yujApZVjgAa/GukpKELN7hikGPimN2kV5r6fGr262lpEA5m6/N6SOuQ3geHhsaVCh9WmvrSm1tvSUUX7LIWr4lIysJoIZQ4i0XMJUt+aS8TC8O1JsDUNocREzUN1AtH9nVBOThUDNiGAPHGEmh3W4v363+IhFGnuLu+jB/T50HKNynvBQcJyA49WZVKbwF7Io/+89Bc0742I7bh3yC9iOteLdXvf2w+vJ8e1B4iyLQ9ngy3cQ17q5bZQc1uS2qYNa6DD85jSSms5Qw1wKmv3VjDVt2JFDjcBG9TFxNdvqq7TedVBwU7VK8faOU4DgC3DcDTO/jW6n3tu1kuHILNnb9fz7A6luBhDBvVEVNPGGNBCP4NmA1Nm8PaCMjwm9lzxJhq6IaIpB4063mr9JZirnoskbdTAagXyzIzWNmjUzjw98hjD8ilxD7bXtMFbuuxIBKZlSn+vOiAoP9cRkkxec9htbBtMmBbCtbCrvWrLDI15OAaIAKL1FUbbMviTHMp+DA/AhznkisuqWS8xhV1B5/j1LiMR/lf1La5SimpUQ3jBSA49m4OzuovwpEoGsy+i069RLCHYV6vGpyNI85IpmJNS5PFd1AyAj4f0/atINU6yibBIR7bRCDeCAfE+UF1sKLnNkXxJRlJfVg5b8NeBJwIhUR73pkMB5HzdQBGhMdOAJdJvn7zVtCiMI4gYdbVvi+ZcUXLl6/b0ERUhze2ryS0mERmZJPNGZSNl8N2ycw8+X0A2PGvVfmuqYrEEGPUYWBmWIw2BS5O68opTGuarvR6B9atEBPfCxgn1NivIohjTJIOFpoVripG4Yr7EFE3xOeAmBBuCajlN5rRXABwurDnHFK3gB3K0XVZzypgmSFK4qnfYMBifkTkgoBhlVHiMoCU35dDas3JQKvsuk5wIO6+uUydk+HHFcXbVBI3JZ3WgAIxBeHnM+d161rZOozQS1OYXDowZ4vmI74Q39I+AVQfyBzXpLtVJSpQaZTOsE8l/WncKwLuwMUPWaMVEEo0ZOahN4DLuhUzRF0R2XCC8Ucz076aptnqeLBlz2aMBXHVqZOLo/kOuqScniH8Ujh4r+pbnIIlE+HiBwpr/abduDAT2cC1Tzja4OfssbvdaGC8BO2cUCFStd1u6Q9jVJ4oT+h5XsxbIUbEwxzdwZP4CGOqbAon8e5B9GR3EoPoxKcVyciRDPZdNA19D8Dzp19oZIKaaKI0yz793FT8WwXprRhh4vlaNryg/JELxjCNA8so9b64rVYSvUTkzHhUJIvEmWiQYGM0S3jBRhncbdbBpWDSGZNWA1yxvdZtCVPWU69T2Ce+9nw1L7G8+/CMnIdRWwnf3dVzu/CGnfKk/S/81BV8+GIkhggm+BVvGzwPzRFogFsKQKMPGbpl1ah/MjqQFCCY+FMgq2h/FTqCoZEFQWgaxU02qMqknlaKPR5aJnb3ejTz972ZUanushkedvms+sxW1Baok1U0btePQvB4FaemArhszzt0eOa3bVnq2k8sIdjTqvmtLeqU6u29Smf5MvRR1f5QQ499T4mIYxYAi7+Ikh2OowI3VikNNuCoTmMzSqu4dUGoXxOm4EDl6x5oKozNFsV424pdVeXvL3naJvyV/X+X+I1qTOj0UYTlha493csRQacBvKHyvkpltMuHhWfrt2TzsaHYnw+DHnkaYx+W9WIryqYe+xE7wn0M0Nq3P9vex0W1MkG+tUCLSCFgCqn60PVa7x8dawnJg37w0q5EWFmVW/6uX4iwSoYETZBfq1hV4zcIzgVSY+DJJciiuY1C7rKIqPqQ7XkaRvLnBgL4IVctehh1Ocqhu8587oN1h0b3YJTRuJwVXKTrjk4EpGxw9ESzTmpxxHAKK/6DBl3+4uhS+ERdEp5njlR94SKEMy44fi16PM46cjUhg7rRTCeEGRzvfZK+XZA8SkVq4ChEpyz5BoVzM6HlTtEk/X+oUQmMW4Su1CVYzp0XYOmxH5oC180H9dAtc0GRkFz+R2r7gA867tdID3q5QA25NXCO7iKmv463t1OinBiK2GSxv+5f261TqOko4/vleP/OhQLzR2d39+73KO4RKRL96nz04fFy2grQEO4X5Jh5ND+Pv3LAihcjxZT2Snwx/+oQ6z3NHlwDUurDR512tcskPnNT7ELl3xNnSuihNE8Hw4AIpmZ2cv1EJYwlsDWsiBcHPCY0DvS3Oxx6zh8mIB6lqvaX+k5AU58pA6JUlYHGrUbc9/4TpIUaIirixPjviJZEQrGnr+YzigBTCDdqOU5wnW8sBdX+KloBI38bbmyXXG5hEVIjdl2xegLqDOgioKo91zABiHG9bzjFUJIyitc/g3/HXeKM45yzANPdq7wGD7GmEmb3IOfhRZUJwYlTvlETL2UW8bvfaqPClD+GulFQHOBaXIXmob/57H2XnGwgS2wqIse/D8fyZ4/td2/U+v9Edn98JGOw5Iwic51DnJ6pdLHhxBpaRRW/QuRUfNyTz7IO2qQMwbLHCJVJbLZbiSnPyF8KTvAtge8ckR1N7ngAH/PWVLf8eLC0cqegGZk7egVgi4CcSfsj/XkzSuKhH+LS8Pv8wzcavLUmcy2W0YhOF/yKL896UQ30E4qJRDHAhwuIMDYUj9D2qa7qj/h/LuKQoAnTtN9TCA5aDKn/AJND+/48V3eSjGNyz+Z6T8BRmM3yu/8peKWgYBfwdlnv/Z2JUdHJyOiT3/iRJDRTZOl8BoA/jk/dZzR4FEanZ5T60Hag0E56Su6Tmiuqm/2hoWKWB/MUHvJuK6zUWZmIU6WvkJpTB+NJeQNRqtqGx1fZk4WvrFXvQ3L+OhYi9bYYAIUdYTlg4We9kqA11Y5A7uLhqVobI3M2Y8UPamVQar1owN31DVmlJxdNFbX/tdWIbOZ20XEH+VJ7ISKdZ9QUCMtW2EYt3xEwxRSvZHD1JK2gWM/Yq0Lg7naEXaE6si7YgXmn6+yGWn1qtXvPayo7DSzY6CctURNE0zw6k6KTDGh04Ci3cbqn9YXXTOtd1TgcKqjtJDG0kPHwz+YzhMnvjfD13hED2GtaPqcXN6/lPz2qK8ZMDJzlTth37+oXiLFb/jTJ5oI32+dJFdS9diZ/7uYA/O2A6Gx6Kkao02fHT+EkeQQ4yGBc5rGqMcgjfoNEBb8v6YAvXx5Twuq5onWxVk8W5pA4u+hpw89L2v8tQSe/5oJ/F4J7Hu5Nw8islMuw4A/Hj35LgFb82b9iFBaqJ+b1mdKiB4klIxY8mTzkP9JwfOKlUnKnplfN0EX5Mf1gUZMRAI6sw3HzsBWwG+rGz22rT04uzk1F7iotHI/8J59ChCrcXpA0PguqOP1yYY1DtcFN7SnnWwuUvF2k/C0ugc87xzTCd+Yit7/pajflRB/zMepbl4bFSDUhpoKVIwLuFlJpg0ALsOuWnoMQJZhTA4+ldWjfJQKSCuhnP4VaO5FIBJeM1qCcB6eGQ0QZFoAiFqTB/EeHZYB1gbf9dQayNWiEOLRa33nBWqAFyUR+QT2jI1BKRNazxfJo5jWUnzcilJCu4ZQ+TBDJPQFBYL5jCtmtTDGVlovBLnIJsZwQ7aOlezjEBeaprT+FiEtzpYwdbP+w6gYBWN0Z7QFbuwlTLiV63GxuF9jliE40xTTzUMwriPVGWsBU4k+wpHOK8rNo0hQTcUN6gp5eEeWA+SEh3uUuvfC4zrUgUybjSgipGFwdqAfENVzcZR6xbLS/wYlUjZjAVCk8JjLvnRdYVlbHJmbao2CEVCWQTcvML7lOchLsNKUdWZ+JXMi5JnubO+8HRYCNzMI0eqm8JmqBhdqLrWqenV+SK3O+2Yl1MXWCCQ9yio23zu1Y5E4cp8MVTRq/K82+LQIQ3P/9XCPnrQOTg0Pf/TpY/s4EP/ZkEh8uBFo7wt2QU5PvGYojOQe4vzhec/NgqQ8buSzXelIjAJB70Vq1BZXe+G1Wjxr9n6WDnv+XeH4alZNW2vl+SUn1zltKUiE1rqInDKLNTwzQ+bVsJXEgD4pnIC4XlQ+50hXGs4iQl7baFU08ClMig9LFVh9an5s63ZVlJI/dHjkZ5lMx77zYJDVp9qJjeeSnFM85AnMWDM69iiAusG50U2NWe1IYXGitwwFQBLDL1t6GLcHQpcNi+2J3ncFTRxaeI6VaXj8u4CvQeDcojUljHPKvn83dgZb4+qvxiUWqFT5MIJId1huPY93GEJn4hE3htTwWXYzTeIYOQVEjLB9+suoS+B4whCnBGPE9+pUuQFxNyF2q0hm6z7jE2tHiU13XUqYYQPx+EDp9wmtLzj1FGV0p5/zxYbpfg6W+FWV4Pc3kkOH+ZGV0gVfxYegAmHxnoTsR4OmtjH/InfLRpqitgOkqI+AleNOC6uYaEvenfLE/YWrAMQJZMsWVcZhlhCX+X0/z6btAKoogcB/P+pKXipBC+7gn0l2O8LXihB5xw7SrCjBKvq01OFgPpRXFWWLHzxc6RB+vUoxMDpiNTzexgCrvp9hTew5kKcL0V0s9O+x46PibziciMB7WN5yfi9x6yfu0xet4rzweLnyc2irDNhYJXovbr8fVvbSe4AVbY5UZ5CqHEv3yzI0njkn1eOf1iEjIJ+cVzwLPyOB2Uuf9D8an9QwQFD73vD+XBwcGr+9PyHQ3rqzz9TWdsIPIDVR0/6FUQvEER/y63V/v1RkRcAGvvZOA3A6ZjY8x+NH037sDx7PK5WQFwFCkIUOgFsom+0QAUqPAB/rg0wWW3adpnOGcI+vv0HQQVOW5nn/+kngU/CHVY8F8nJaIW57m8ISkCDBrxeuhPN1Gf8eLPMwbA/yiPyfdu/loJscAMtUN56KDC3zvNH8SfGezrFZs//158cruFQnPznT94pmAUamfW//Zf0ruBcv1+m8xgxDgGxawgI4WPwsKDf5ot/rEMEd/31Mn2EVc61OVLGst5lSLeyLCZEF2HinTEhhiEqvnvPHsx8xfbv//e9enuylIPZGjUDg0XxeVLSpC38f/5XujRIS3j05v98LyiNZWFJekAimCVjh39GB+8YIkidFu+LR0JoIvDv6GQ70oOicpaszhQKz7Ljzu7pf6lx9//LpdG4812MRaf9xgFuo6yhtb+gxMjTu9kjgm92+5/ds2BWSo0X0jS4uOIBkI4In5yifitsBNeclPNx+Idu+yMr6tpKLQAnnGM7wjtGaI6eg4Krpx/33GLyIZbgEnwzFNuTlSj25rDipFNWfwrbbNg7TMChD3XShLX48dDhScLnnNWzkPhVqP1XS2hj9mQCBRGxQh4FV8iqCWYzj0OR4y76jhvjBq4dft4dEOs66sem3IKZIQzKKeDxx1yudIiw0C1Hl0Q42+rtIklBr3nDRs0xf12jTRQFmvGywhVjXwVgJdiFP4af0yQDAbxKHzLH0EQAPM//rA+rM88hQpOwsnKh5xhiTSFmicEC2BvE6nH0hxKNJ+Q6BrxNjhNB80a/GZ4TOBo/MUTW+IIty6GAROOjHOIUjPa+Xw/pgwcrL9N2bhjT8fyPl+inbf8X2j/TW3/S+/JJsiBHvx03oLhuYBtZxCjXe4I8umYBIWEJ25l0UuVHwbl0Ask+R4F0QhxVIkmcAkgBWEs5SwpIbLQp3h5CO80YpdA5DpJyBESj9WgUOYmxVPKwCNL7C9TilK+0KuCspZxD4zhVCofvBuAJwB0Gq/+GoYLoRwoKh5dVB58JHoMF3GTQ1wHUSp2Ihy08E7rcij0q1tKXq8g/bCY77JeAQFZaybd7LE7DoNp7dctAW+qBPr10oTe152OKSEal/unHvu46CoAg7veRnaroU9xQhoyYTNec4E87r7Z67bKe0FUQ/FS/b470rwJDMXFY88QzVVRsjMYOoUc1ilcxu5AWM+LQ3TBaOmNkxRBhfTHob2uwqobGoglptkfe6yvBe50JXgDgRl5jN6rsYljpE6cSMcczXWiu4snwlfugz5OA/NPGyFEIXuqDirMpkBRO4rc1ltQj9CYm3CJuRgbhUKB6kZewEcLjHP7yXUkQ3WwD5H3+Rav5SUseJLrHYgpLh13oXpJPWSoSLG6rAJSWp8yahD92swmfupqRe8ulrdfbopLPl6dmhCX4I0yNeuQ/7EOBnXabPP9WX6uxJuSeE+TrcD4E/3U4R1SSgWOafkdoGnOJ9snjQQ184wBOAs73EX5IgL1LRdpw2CGl44MF6jxbpAOVu9Y9OnQCYF4YvoMZD3kK2Vb4LAxUM6C0/KWFaEYtGIJM4+OO6que6qtG1UBfG2CkJFyd2w1QWlODjxkYZK6tGohpUDwLN1LmGjtHAakZPzzfBWZPYSg1D9vHalR2VFF8iHWmeJtap31X0nJ/KpwdlXwGFpMy49UhCuVSvKV6YMjWvOMCcdMoQ1lxrxVbbDbNkJMv7PTGhvhbP4qeoMOSSVSbIi1uO9pbE/2mQxpnUZwkdx0SDWQEj3zVIQ/LE9dhiFancBOv9eWdnnA6FSK8bzcjo1OLF4BZQk5m0KnQYHZd9Dt4HEzP/ffcOhDIFPRJbbo1JmIa6yFRlAIwNyDdU1EhnidBBjGvhuSzwctTfVhwddVLBSCE7ARv1A8Lug3sKVyAHQBzjyxF4HbNctgU7jRYb/ifBaOHX7zZl6qt7Cm9cihZ8H49otRTunr/2aiWmmWpiDLKluFeJWWY5pfpGnlzAXDlxRLKLaQj45XXPWBvANmvB0y4985K6on2T9m5QBg2kzwTvUe/99JxzgVa6qTWGCxm8V4f5NC6+EykeVXmGZvZA8t6wO0eqY9LmMF2hEXcurxUJEmdOm7CgmvEtWQBCGPK42zoG7FwHXW52mm/0fNfuzSt6Z+i9s7GNbTh0PQzc8YbIMOmU9olwTphF9JhYfXId5tE6a5rsng4pARMJ42WNVPMTtjcATRZlOLxMmq5rLa7eq7xbwFczurspOYZMVAveJ2zMmUGuJpVY2whYWqYTjLBhmaYT5c+pEXQxIOtqckA+nTPNk+W0abH/HRZ1VxWS11EKJW59dzSFmnKWxvE8YremOpEoaYR7VJIa4fUNqhP2He3Wm/NgCXtTTjPBxStm2j5TneWUQdG3uaFfbnMEapsysCrhFIg7bWoxCfLdDID5LbscJaX2WOn/k5vPXyxQK8z+PZ3nrgO2DWK8NsP54FT1Z5HNmwdjV8DuWR3TBGZ8IQp4wDqNcQZryALuyG9N8/QQQpZs0Qt3ty61U63ZV0torHUGbh5o5JjTo0WURjD+OH5Vy0R7efNlikyk+R6wA8Sn4/Ld9ziLoZtI/D8LyyBi+/cSWJuaLTjZ7gPA+YUpqLPFh5AjOxgSUuJFBOYmQ+NInz9jkdPeTVrDn7zfgdD3fCUV2Jn4eFYaW50sTdwxPCDHX6gndffCoZY6/XdkLOKUTaz53/qUncPMVv41CFz34inKdA7urCLLNOtQRn4P9I4W6DAjz2ngqNx26mo6OCYJCaaO5YSQccZ4cWLJAa/F0SJ2ipNwmmmOFoPvlnBOj9v2xGZGCxNx4cB2gYCtVudAoEFFOJ92tED/84gxLIp3HTIOu/FIbPfy5ACP/6io4BAb7AUdueTQQ3P/7VLY/z45tinDqn7mMZX09E17vZVT2H4ERuyLmj24KU3V33PFvQ0e9dqI0fH4bHeM7UKh3Myudq//nNI5b8G2xoGa1EK5ajS7pxW3oCu9+6+h+6tNBqE61kuKw0PTcv1xrDGJx0RlZk3z7jTjv5Pp755uvVGByMSdDbl+jYF/XMZzZ6vbICZLsrK2PQtX+80Hs4V7YZ6Ctp92vSqGzxfTXbaF2ZDDWvAwkdjSjzBOFIl9C63q6bwFic5YI14QzoIXyJaxTVbkXx5aZF02jGNHto7Jw9FBVnpiYgqMl7AQlQTmYlBTp7WuJo5ZFgGALJ7Q7IG3fxWX0MBoMfVzA1g3rhPr5A4QnQOyMxV454YHxtgzhkvhAPrvORHm2Yr4ZnwIyga5ZGwZGTGNsdt92UUilPgH5h1cLOvRLjwhXoTPdezup+2rkxNNGZ5ArzpAssvqY6ii1Cuw28EDX6rI9WERWicdIUW7uvDrtD5gm50tCiQ1G2lsKF6HpgETbDYGtNPjRIbK13VIdG5bnQ1sHRpvdsKmQxAqvGgI2g+36almeUNHXzqBwen+L/n/4dbwTWBg85SE7hWvNvrWwrFCIA30L+4TsNGT6HhdPJNEWWed+5fF/pZZ+lgymOpwbpLAQbxpkugysmuaRn5yTGsTE06xoEow1ArJ6vyli3C8Ed3yrKEMG910O5hbHJ1JAR70G1DK2FHM5iLoIeLD42e/xJzZ9yVBkMiz79+OGeQQHlwcKr+8vzVxujAiUmzAl1QbDNMxlW9Sul+hEdZ5Xr2/Qgr99/4K12QflC/ZjciDZvVpJD81deqcJCl59+3tLI8w6Oprh4r8D3fCd0PmpAA4oT955Ii1p5/e0AMCYqeb3MKUIF7ZaQ9X1byKqe887+p3+pm+kicrXzBJ2crXqJHqxj8/A1NEUBBdShZoL88/5GTPaACklWTMeCFUw3BUYokDiiB0jzA3W+KWEyG4rNR8oK8tPr8eFQZUkNM7XtO7ThqqQyeODV4GMa929FEEkjewMupVBtyYI3QtxphKmRDk3Bq/fb8yyZVwlRU233qBM2K0HIo3DCUsCYYgXlXVeskJlxnnkJNlbyumqmzKAaT6yzspJJixj+KgR2bF5fjt3vHsp121O/GQroCJV8qURm5KtbjtzsMbeoJjIoM3P6TvC53r+rcoAZ9/IGBSgEs8WAMMEaYTAjY8+TJm2dunYzKvTopIQuUifgejlhWuen50bgyZa08GuiTy3e4AFt5wQU0yvoCLuCLSslmv4I/2gJDeRljFeCbJcSLq1SWDKDdnHD+vqUWsEnsmaVtXzCztIpvHjloOhwsHT02j7cONo+3nr/TtDUGb1fZEpj8HzBCE4GLX+cQW+L5d6ZqLu6glajv/VYjJlgd3RUvSyThQO6QgwP1x91uQwetZkSeLJDHC+SBW66lhMu11pcD6M6bzV57G7e70ZNBnlO/J3o+6712BaK/2hPAmqj6aUmikLG7qMorDQ8LGEf1zitqwNw5YmrZGSJaOXULPH+UxiVOd4M8acyZFVMXWavzMt00G8EvazAiPhggawFkGKBpgY9lf0BHnxT9vTOMNmknoOf7AwcRGc9cBPt6r/VwQFPh8oQiijMhFXNMLI7UrvB6QyWjSGEgjadpQ+9jHgGSyTFI1hvJVIDy2zoODlsBZeKh9xfIMWHJWOkKYDGhq6jIXgbVa1YD/nfdZKoBILVChJfNtpzHRB3TXCBY/hCHkhepXdAqdon4ZyB9CVI4Lig6GtwOU49wsWR00ZXFkim08HBTN8AmLp7WeS0bYOyrNkNNJo4uqZYylEexnF1pf8bTNI/D6waBDUTeIJX1ptGmocHoeV3rS+iOpyKdq4L340/xj377fq8d6+BJ/5t++36v/eWA/ssB/f0B/f1B/Rda/0W33XVf+wP3tTtw3p2BfnaMftbbdqKg1xfUE+gjbrYCVU2vD+lL9DFXlaRlQKLPojOL3FaN9BWg1YWIVwydIuuWlAhicGzcdrAgtd8CsfaEJ1nzeWHZyWavGc+H9H02hVKQp5M+hRKQJpmtLYXSjhPkrnN5QLkE2y90EN13kyTFfEoIStHe7iWlkuzGKdvpMADpiVogMorezj5z6ZATlIk58padZMGszMHBZLPvKAakVvzYJUYuMUXUo4oYtgf0EBBnkRIi+mmlDYeS2gG42JTs6QPxAG871PAU6OMeo2SKVUJOUJe/67MiqUfYOsQWaKwPKfh9gUGgpL2WMIM7rkJpjlyFrWG/4N7737HEAY4mRPhQvaADg9wr8TSDTduK44ipfRakd8LKDX6MKKcP++KGpRXOkiFT0dYirdtdhUj5fpB//aaDSIp0zkxFBZBrBLEpDyWCadJac56o8QDyTm4ic0WJc0hX78Z0kkXasVzWCfmbr2Gr8uMmIptWM0+5GTwbsFhngrV/XMZjeaBdP7+AiYh2PR9GkA9/Xrux41R8GKV5mNxD88ddN0CW6AWVyY/5M1fVDyqZgxnnukgBOJlAohUm9DWbHguuosuYleH/ndZQ4P8rdishxnUbeRaIZFWk4Pw/PoGtC1B8hVhrsWkQZoWy0kVpiOBA/FiKQUvpoPdfcV3p7B3CyPb8B2OVFcomPxvmrPpRiupyqLLwknhS8vJkFUF8EBfk4KD9+7yqoIESnstNxkxxAhfaYYBC5+CHwUmQiHMw5cAASK7h5NNWP+Z19dDYj5HPl5zd6KvSyetyOqgFIRQHUW6rv5iixlDUEJYGfxJrIiroonUowMSN97PhEKoYxD85RDCttXxK7LDh7BlT+mZcR8dxl9Dy/O+GtBAykkn4e6wzU83zvx5XS6tiYVdpVSy8KsGnidhb2BWpLexNJ/ct6I3UPP+fl+ttGa37DiUKckL5H0YGXYNtqgfbgwEhnUMlNA/p4OILjAGlJiLr1l1oQjFdG4ArQUvLShEFp5IOnlZlPJ2KUt41ldQ6qsgP4auSG7acwj8U6tIPoOWMVesAHHhvSNh03dC29RniTodEBpXaIPlcNa12h5WGJJpTiuj4VDBZI/2cDgs93zOEAzR2WZ0kSylSIGJMkaIxGHHw/EeLFMllf39ETfEn3TJUDDY8ChPdcQthOAHkraZV64HI3ukIrJ/A5mSR1uVhg4N5oy/hYXjfboVMy27LmtVAizEseLpDiiprGu9VRysPw72lyPFs1941C10ZcwYaQjXNyoDJVHc6rR1ivpsdMaaj4YEWLZqOR7MwLiEHAoMGvzQ1VMCORkqHFc2StcR5poxqfzXx3F2nBjDkkfy5Q94UvmJsBb58NfV4/sG4+pj02YJD4XumD1N6/hWycBtAfP/F+NGqNJAcTWg7P3Ic0Hw6aLDhrXljaroOAB/2YkU6++PFihigvO/Sg5xWnumX4+QVTBKDV9Cm5FOQq6UgiwzhbXtkfy20gCYEROLM7a4iZvPv20qfDDMA0iyYl8aH5vnWuFbzl55STgdl7ZxmgtlRNWGU7t4ZEtIVWyO+SzHIw9BifMT9+sEBUhvxStx0yICXqipdEngyWWhdDnoys5q3hLD6HSLRTztlmCSjSoPPckBdBISf1mQ1G/Wy28ljAMKg0yGR5z8YPqqZ7olJTxv/GofmxN18fNNBm3jEy7QuNh0SfDVBdbzhkGF8YG/3/oAorNP0hBJA7gyo8LIEH9a2Q9xrMq5b0xniD89fM2gc60/JUgKnwg9We2cn1IgW7YQsRaI7bJIB4Uz07BpEoMe2hoJRUdgKhUYAh6l7MRVkmDc25Lg60UCe6j9hpl+sbnCgAYhvJotcip3w41Eeya72tlMbz4TF3xiKfNiljyQFVXkVpbRj316kBT7SW4YSzRhNuV+a3jSpJg3GS8+/jj9EFqgtNgzrbhv4NS7hQgYJKjv7Ozu/Oqt//tN78R5Wokw7fBp/ea+OCqy4wFkKyOVE+ZOwPGZ5hqyMP3z1k1ykeFvHc54gCzeCnnZ6/fb9eg0gD7LHa/n9+7F6Ihark9nyPWlC9Qtxd/rv/2CnI9f7fiOSV3l3RH73fh1JYCTvM5j+6f16U3nO8bx7m+/3nv/0w5fsi3/6/AdgHu10eAfJEwc51g5IvAwcX5Xjm5LabfgLcmLCarnSosq0bsI1XX1Z8PJQ8W9AFGdVtaP/Y1eRs+TlqlEPjWWjc4LH6zZj681eK1RAgGS/JfYE0Cj11pBVi0EdNPr5RIfO6pOF1J0kg7g15vCM6kOaCKbxvNzZrZPFXJ54CWcnERG7XATb69W+QhY6Q79+ZZJpInG42usgsaaMJSbp9ps9/ze9I3k5rRGDu3dwX+L5u+9KFur5Gxbzp5XXvDfCJEp7KjAZkEu0IRJ9PHKMgq1B4bOFepiXpzTM3MdHz7ZcxzYupUYLSeGfjura1+SPkqCamleZSImPUWTsJBZJeNdsUdjqKY+TSX5M6CD4ERfMyA2AoNE5xuYpAzLT8/gXEZSeZ5CCQsGUlh1VRrvrmkpvwnWMSM7i1PPPI+1VMCuzOjkPwFopK+J8N7rS/k2Fzr9oG/r0qMSBegPpmeZIEtSiw19mEBqHUlkw8FlylrEUS1ovqmQ3tKjKjxh4K6PdC0yhKMI9XGUsAoCzXN2fCKEFueOslqKujoKm5TyjyAfkGoJ0KkzpWcbCeF5Fe3e6FKxfYdgo/D7HZEZLjJ7YWZy2hKi/tulPCzoItlEqsdWgTgW3S4Ap45mQlQh/swzf6owDNixPJyoLpagnSRyg8hM39WpKL9YiXw3Yv//7H3d/5YSLynLyaf2qqfP66vtvdhS6AKXDf/XHb/a22Ysg+UP11Vdf8u//lR3HO/ukAfV+cL2Tn+wc0fA5HgJpLBCuwmpEtLGn3SbP3/7667/8/jsB2Lxff/PXztyrpvn5S8AR44hdDU3kUSAomrrMWveBQjlUiF0v3u0oYI1d7gDlMtWst4gdeFH9oDd6Xv0Sx3HlHXVqTNRS0ATkNNvOo66i4rRqsygx1Pu4q6ZIajDrWweV80JuDegRLWeeidsDCqUoeXa4OSCFBfXegIwi1PDs9eMgwZYr0YMi/MDsVobrisiOEmljtBGm4AhWZJ1tVeNbSjhBYHR7sOyjqnolHQQkDGzi7up1q4Ckd5jX3JCJ0hSJoV+yRKq6SHDD/ccxqkVFvTTKt2jqeP73i+gfF/aX2x3+frTDaEFf0VLXFS9zXfE7XFe8gOgyzpfrZsHtxc3t3R9hOKSal92HYyq7bJaX8Y95tjGsdV2LgqLSEfIP6izMVyqp615oMGMGIRIb0TKgeOiAWzC1WkAJyRkYWl9MZOHFtinJp9vtLyBwR4s6qQHDE5MTaIpaaZUIgzKp08zoR+TR1fYXD+cwS1+mFrC1j/IylGfo96VKMiPb4Jr1c+cVS0SvaYIcj4F1Y8+QIlLGiOOAkUe0qyABjWoTwCDIkLeMZ3vLKoei4Q5ZlzWEHAADEoF4muytq7KqJ5g9q1vOq3cDb+WeSu1QSJIs5RmfaiI0Ng2AoAx5BDXULeYgz9numqN5Nme7LvUZ291wNKPhHgdXSBSHQUV7NNXFv30x+QKNlh++/h3oPx7jEER8bHLoby/Um4rs4aiSnoUfjGmpCN6jMZ1pUzQ52pVCsT4rM6q51jRXDeehfsstA1ZLeNgIdxwcgBWw+LnIAatKtmxcvSNUdduIRiwZceW9dGggE5j7vCjy/AfDRzXn/npAh5DrYXkePkWr4/nPluinOak3pIzEt4ai8w6ABcrQ2XbokNFhKP3aqUR7D/cNamlLeuY+tjnFHxxqHe4395k6Sp7/8TI9Ned1EWWmoirjYGB0KKHnPxw7cvTR0Q6t5O4TaKn70bXHNqdwDW+dKzEkcg8OdVTT855LJ50MDGqQuIcyHdMjuvP8FwuVOySj9xcd4H7iLZuo68K11P3E22NHB40Cnh3gOSWhe9A0Rzb9P3ZpWRyslF7zPiyvsMN5F5bX9kguS8/ff/cD3+UgnoGjpYgD4O5d8qCBp2pQz9LTerNAy/2smLZXf/leh3v+u5+WkgPe+7TE4/vqnY/D97v7roeNcA13DmnebXNvy57MGBWuxc9xiFuR/Kyd6XOzr9hQ33apelN+KIaperVWBJ5JuEni9v3YKc7qtFdGCdr+oHZbZ0iaDwY1oeqQdLo30OroGk3QeubWQm8blF8bP1t7y60M3d11alA21rAc/MYkdz5PfR3tBfSYkQHsEl/jw0bDIkduDgATuzV9hrV4WWljVucoUtIwbpyGCIybkXXdJjBG372++E7qp8NqGyQqnu3m24M6CkgI3CveoBJhKKaYK1KX4tGgYhMpkUeiHGZQBsRv+n4SfsKHz9vq/R0cvv4SiuFOnmbxMKVxo+lm/lVpqMqkf+zUaZgJgiSmTdUA+THY2C3X0RO3Ur+xpRJ3kCmjPb41LCdb3H1zqNB4JN1kzKijH5v7vhq+Ztr4us/VKCELxmBHCqedhrKb+lkhDWECCb2WcUUZZ9NE4Du+N6AolHfD891PEicGUnnkVMiaCoSQuK82nWoQURp6m2oWqEThPpZO37m+7ngx6KPBBsBAqdSTj2apJ/R6hjmBJ+fLGhJdMWy3Ai8JIXAN2ILVMmmTzhN+pLy8lzvN18pEz3GafhhUtNM2yafF3RIQMysgMagznDAxfVy9J8/faOTiGHgaSIxO+67ImO4uWQ6XnZVSBALeRZZXcRQTKo83Sv9sOFYeLlacimx7VEs5TR6PKrVek/HOlNvkmsUfXb3dZS+6TTvsxdUuU/U3vZadXkv3qFe9o146Wrr9vOz1s987ar931H7vqL3eUbuOlu5RO93Hkc2jYt9uwrriPj/3Mbhso+uO9n5blqw7ebxLEa32BM7mXWreGmK9Bk/N7+NEDNJnExXjl5goHs/F9gj19fdlnPLyxPNXgNga8lKxAk55SK9BI+7X7SZCyNNN1xUnNgOzkVFOyhVsS9oKW38B5XX7eS0ix1bf1wJy7MZ3+XBcT32I3rhW+yUu6E59ipsqMVkci6Cu8tLEj7imcZBKruujbxn7KkpJoKK9VIQx/yHLs49/2mDUPy/qcvk4zSi18AImbmz+9yWJtCF+ohz2KiW5d3GEgdgkscl/+y/reikyZnEMSyg5hJq/Pf/lOx57hDvtZTl8bVrhZTl8e8TPDg7ff3rPvjAUqyosEjEXyYpF9lrlSFd02WqMw+vWb6oW3u6yu1IUjpD+MCmuzg7vdJXs309HxYxsqYbhsKMrjguALUv4jyeK2lN6fu+qEOcBcgrAlj1kRSwCcdOlhJ/3I5To4AWnd95vu90llrVCHn2yWEv8dIidFeDA63liclgO6ja0sW3r86V0AZkMdV4MqacAyjNN4rTDp/np4AFxFgdFOh8ild17lwMxN0dTfY4eg5pEqX6wQJsBfscOfBq9Ew0+Blg/EckvnArzMbxc7gDCECinqghh/50oeRdem30QYstNyzgcHALGAcZjGxxdkHcPA8a4gVeLdNvGhjiYpyP3rQ5DEKtWH8/ljRy0a55prPfdoeH4bOggmmnsexlWBsCoQO4t9bUqAhlT1x/UTdKX86W+6wg58eCBmeqDXw5g2ZCy8VoH76/Vbps/GVRWHCKI7rrElTd8JvN36n5/Z77UuwnymSjF7nyp2atVdg+XoasaGvFDz9891C1mbZWwQcfiunKtYcQGWlwsh9lrmnroswMSz79ikmWDc+KWzZXdkCHi8rhpC7H2a67CZVdMJm2Q3DE4mR2L3f1xDmxYmhaowEn6bNuWCtr5wyTXcJI7g1I8+IshZumjWSyLUX5qpeH5Txf20bZ+N6TbuBZkcThyUlNNrxMLe2tFn48ewBedmHv+k0U9dCckF1u43ryZ2l8t5hYfvDit4vnPFvfSnaZGlWEQ6mSX/SF1NJyorAOl5jn+MHhQnOZaffDGWiXPH7xas6dWMDg2THXz9n4zdICDZr4v93x/wfEdgnaXpuZbk4UQwWzklLai5+8u22PXhFjiEPMZDQ4vSluurdscnFuawsXh+yONkbml6WOJ763RNW/l6yHtMK/kXpZXYuT6Gh1tsI/307YPTh8U/OlbBQ5VIPTOy1SXJ+sbWjf0NSk2rrcfAOPjpZTjUhpKpEI3fkJIcaX9iVk3CNR1025EkFWKknXUAZgkL4GivGUMb9zpRJvOGCzIvHzkVkLIPNhiK7UHbjV0k2mdh24dbZkorftuLfOaXrtVKKv8bR2XQm95TYp0dfCL4YM1PwOvRBac6AM+GT6ATtLRfzqsj9TvYExp3c+GdcUx8KOLUF+V41ZgTKTFnjlE1E89RNRPe4gYjZ0hYkhoiKzrxtaTy/Bcl4EGJOGBPrf5G8HljN/t2Tc6rcbpuwfQ+eFywxKHLstVdieTVRjnHQmUEil4lo4Ers4toYIslGyQBC+g091aR4T95dNp9xDjPN1D8ESOQ4wLuIskKUEeur7EIz4flafF3taYHE647VIg1HEiUIGzLFZKi72HC5XGz4fQ0EcLzqeVRs+nleB898e14GyLVNJi78ECFTjT7UEdOMmINC327gxLoevm9KJsvInNcFRlA0M6OP4W6OCAUzobhk5nyHdF3aFtiLDHeSS7h7iGNojwiyc+Kt3zmBy6eTgib6eWJwu0jKlmUYc09awRbf0sr4FHG3hgKMHmCRTDNR5Zg+pIFwG0SRsPbNV6mvZ1no7oAFY52gpgTkTx8eOeLhK/dNqQx8tsk4XRJ6zIGEL3bSUA1e71Bo1I8mk2EglimPG209MFGp7vL+xDCzs3OSlEr/9JIZD0zi5KlEdxBeZvE8LrlS22XFOqS7fQ8x+OHdmGCMe0AiS0qqV42dNSq73rGpQIWeSGjmrLK4Z1mrPf7uqU0ujBKW2O/awr1STzjPAkx8Se/3j86OYaFug1V7Pd1aOEGPuRjCk1PXk9pVyySQ1bfCQ2GH7BraKhdcehtVDcXMxeVwxMq3EoHc9XSTz//uAxzVmHVZoz9y4slXzsulE8PDhSkWq610PHxZtix+Cwjx4eHLZeczVfjOgRyejYBYGG5z9Z2EdzWYtVmyvrTUKt6uhFOSYh88jhMWpqNVfxcV/LOQlCs+ffdWs353zUldNSCUW5cmm14Skq4WNTVGI+3d4wPyRO3rEHpAq27VloVKs5XW/NgfIOZq45gxpNH72HSyUiTQ8D8uFX2XDU9psdr1Kz1A6djeTN2T5xyXcGTreD/HtO/eaEQwrNGXtvIihzwBSyn1JvokecLyj5NpSe9pQS17z6/7H3FtBxG13gr5jBDXMctoNeMoTapE0pZaZ0K+/K9sZLWXCcQsrcpszMlJSZmZnblJnblDnvG2lB0s6Mtt/3P+9/3nk9Omm9c39Dd+7cGUmjmVgyb53hDGHR+VlbIIB1ynVC6NHHtZkCziQt0Hr5j+tkFtXb5/HmeAzt1csYwok6xRCv7o6N9upuDt27S5wDmQNH8OUtFRehE29p8Rh6CCph1r5G2NKUCcgQVJMGegiqQdFDUBXFFgoyBDlj+hhqiaqUIgCnkCWAzPDLMdDetkygvYl9khJwKb4dN5Y0uxzQzCpUeeMI38XKasuhAIecDFs6ABQc6tlZPkTX2isDEm6fnlG+7xnq3rOivxCzn68OqQ0H22ZDcOvLoRHu8MKKrHWwIogyEi6yoiFkVuWGeWRgTai1PeFwmMBKbhxMUjnqAiQKjWtJxtRKwBKAbrDZsJEyx6LFVtYYOSrjXmShe30L3WtLRnkkZipbOvY3kzYnIoSutFEpQJsACK2kGyECV7KwmJZgtFuQNdPgWyQzb1vdKJQUtDsyqlWkZpTU+Xah7FrwpFWUiX4U6Ee+SeHrDDM8h9T67MFjWKlELgMeCUT7gnY5x6DlELt1imF265TD7KMit0+JQghBxqiIVq4oIczcU4k0on/bElxXsQlYI1iHCEQ7E4ZtW2OQUivjqUhxjXVN9kUt1U3yxYB9+ScGMyEHBuuMpQEgG82OhAlKHmAURFaxTViKlq6HQwTdOdOMe1smkbY+ZHOcRu+RgFYZAZXABpySyKquR2Rnb8caDxW5dAmPbYlGekX2F41We9ZEs2WgfggRqCAiRdjYWJZZJRmHkNkfSoFx3tNCZipbWAHTti2AOXVbAjme3A3YeXrmBfHcCug0AoSDQkCCgTogqVglg4TDDNsKt4rjFSTyJY/lTQkIQM6wCLBB1BLU9PlGDGRlOxYDAM3iEoDWFAAwh2cLLB14JLEeI5G2SzMMJgHNAhVA3KMtqO5rBs3KUt8kmKRGf+NwlFXkRhwBNIhNwtLUeCyBVhnMK5QkViSPbqyvNqxvK+2Sj0aKgcbRUojaHVJLuWixVeaxaLFVcFcntKbyVrRSF8nmwMG0mWQitsK+l9Ds8HRpG0a1/NP6Vdn21iw/cvAEjKwBSncjqWxyUlkGdjvs7zNzYFMf+6SwXKynvI38RBhVPowi1mPfRc1yQsuNvur7nbgZcyZqHx+Yr5svFWImnvcWZ4YT7wM3b+Uvk4OQwtRJl4oSwdGlx2CVnVwqXe0fxarc8U7HxfJW2tWcBSPZmzAhlfWhYI1eprwZNrmgiL3QNwfJ0x8sZTsFAXpzHufmIFliiFJejV4Cq0/raxZYRj5UKbMmGGU/f7AOYav8OREGYhUPdo+oS58O0JtgM5SD9ZU6SHiVq6Q3c28hO5NmMgXJ2pcrZTwZzuGzBetng1FYA/ty0GwrnDdbN5bKJJOpECRXP6yU6SQo5s1zmpsykslCTyLdC+2rdbGl3KeiWXwRzHTeTGfAt0EB32RdLL4RTbDCDjwM9GlECAdtxArnzdbVS8HCijZInhMgELbjAaYVvDGHdbwmFOlbttZ6ytZam5CbScdhTTW+lsG60ZzZZcQKfgNOmcKWCFTfr2ZeBlZqm8EaVnkVAySpVgTnfZWZtTYdqkwH/lm0ynxgKjwabCCZDEexBmh9RdGXTGVgU4ImFIn1cCXQz8NBMJiHq2DYEdGi0tYz/ZrkpiFAmAprbKDE+tpKCK7B2hpD9VdjwRCPXlvfiCdPmNuAgkjF2GB9iolAawLJvI5ZTxn0tatgfQYThBlMTRWCcP35cqi2DUI1UlNR2MypxgAgXt6lWscnTXgDgIKwuaID9Obscp5Z6EwNh5RyG1eDYFWVTRqpDCQnPATTZwnCjlIWA5uV+VCwsaxM+WcIG/V9KGSGfreH2Z4E3N59KKhGexIwXWEZRDowFWAZ2IBuM97qu4bOrJHuLto7mkFyrActZdyMRLEOJ2vE40nT+m4Za/aZnPU8AG/2NRBMuSUI61MzWTMNv/Xyw2Cut4JhpxyAMhJRuDHWQcImJ04Sr17vVAeqXsR8aAIEwg7lgPGzby8Ds2+b8a0YrC/hIVTFIL3J9cgsnUmBE2/MXAHRjnXSsD5Vpf3sA07CBre0mUGl6MvBhvsq51VTixtLZQq5TDraE0XOrKbjImA7UhX2VRSUhHUkJ4n1Z2XQX6O1HFyjZQ6frZHuNBPdvvM2GAfNtsJhVV394tpP1XASpmoniX124t4nB//4BsXCnrO4WeykzzopGj/p8yCwSZ+FYEdp8F9wBBO8q9aDwjyKC8X6tRQ41sZI2gerhf38GpoulWI6jsbaeQmGzbJ8uVLmIThnP2awVtyCJezOn03NsOrVxHF+pQCjrdqU/jcZjuA7W+XQcd/OBiVhU2Qn6SxfEwqqr4S+z8vhJNQdOMi6Mm8J1Jt5C+w5HlQ9gPRVTwvkkd9kLxjLwqaLUyBYHTZey8F8eZXDzpTAfhS+M6UaCF4BC4KN6hMgHPb2LGUmk0WYymbCqEo+cwu5ojnfETALj3cZybyLnwjjvSUd6IKMbDaR7p7oCUv6TqtrINjsswThW9BIdRr+uXkg2B2LBfnehddS0OHUsHcPh78QrQeFjl1OFPtICOw6UddbQQeIfcuaLJ2pa6eWTFof206GEjVh0ITsxWu2+eShCYHvd0uAfaRZvtk+KHb+/MnTG6NbbBvddIutFk2d7nqTW9LaWG+C9ibR1pffYO2DS63234gb94k1pJ/VliBsD7f+hs0pfCiYvy5TWINIJo24gbg/8wdhluMA/XQW9+2gNRBCsXFfxSZS1p53PoqtoaCKLVHY23qwwZDfbb2Xgd3W2wx2KAX7C8Jb0A+DubsK5lc936dyXqaUW0stY4801ZUu7t9effgNk0mju5iGmZYPBW3rEoWdiPdmkonOXAI21/LlYJOUKoe9EbJOqbS3ifEbTBAorP1dqHOe14yksNZpn6Xpu5gBgiFL1+s3964ezuk3/YWTsLm3k8Teii9NpA37QVYfrNZ1sbC3bw4W1vXqYmEmvrSemVINBHPEJQjrOJYaCdj7KCwD84s2gx1hAOI7ia+B4BWzIOxIam/CkoS6vCkIEFv8nhXRPthUAA/Bil+CfHODmeAUCOR7M1PisH20p5heUTTSUegqgDpI2PS6TMJHw3pQ2PTahWIHgp5cKlrwrDCADQQwDjYQVDlstt25TJ9Zx4tqGAfLtsphu3J3LgN71IllYF3ZZrxZdbiR0v6HpQ1gVqKFngdU3pj4p3VoGjYkeGnnaDkdB2KH9SoM88r1oDAjdqF15Q83FKi6yptR1KfcWhr2KNRLYycZlT2tfQw7W0ibmX6/uQgEg81FKhi+l2QLvu+mbQZfQ7DKOw+zCD8Mag1lDObKx6HJ0v8mQQns8Ah2Eo8iVl+1oUCf5XL/MF5lvdw0RDzfRU0OFu+Qk6kwYhDy5WDjWpXzXdNXRfFWmUzB5hZYBmq5FuPNyvU80tpmpXpOnK9P8eLYj0o8NGy6VzcPe5hay2NnURbuN2ergWBzthLkX/lw1MjnE/mCkYZNO+rmoWZXwzudQK2eXCj2Ft3G/W7RaynY/UuZwrtPAIVgt4N+GNTzlzH/Svq+dSpT/kn5PoOupZAZ+hlyVzIBtSc8BDPkEoQdHLqMZAysJ4A9QfEHYXcqDhCrWJvzU2wtBVNsmcLe9pj9RiZdM7mC3fbASdg81Elia2uDfrWtpWC1LVPYbmdDsI7uh8G6XQXDDr5mMZepZx0PjINNGqscdtps5tIJMxyN1HGbWEZ9SwjhoCWscPjWN7qTpu/61VoK1r/KlJWL+43hRBiHdTZxcASGnx8pQVg/4tjeHe9HoCCsng4QX4VMIQ/zXngIWk8bwrZkvCtp5GE3sz4UbJAvU6Vn7tbecNW/p+PRgOvHFCSMsZAyh+0x5UN3/HoMjIP1mCqHt6fyWT9+z1sdoO9NhIPFjhTVg4b8Rgo4CXPaThK5hsMJ1VXCkO8SFzgJu+13kjBVQqtTgXHVCUHmPNBGDForsOtrcC8LG54rrK9XgoGwVwEOsG5zg9Te7XfMWMLvjtDLwO4IbQafVWcO9gwGy0CzshhsD45lUilwMKPv9+dQEDYiOEBs5ygd5VXHRA9OwjqHk4Q1fBMKxk7SSqDfJA2CwSZpFQw7gsUy3X3Qbwp8KJhTK1M+hhA38z1mEva2wR+EG0IFxFp7LJnI+lm7l4FZu83gtWptZ7jSG2TtNe6l8B7GZpD7VtTBOrf49LJ4c+wxCt3+35tAMKg5ljG8dfQYKTNpQu8E/UGodVRB7MSxM5nJwOqKh2ATxxKENZDORCFtwu6MfChYtytTWONH3IhhGWjlPHcI8HSgdxHjaxlvkUe5EKOy4Y21U2czQlgqQrTHiBspIxdvakYlA3ZlcluHQ1gqZSWZMXjOU5oEOAygjjEGTsJmK04Se8NbBWH3lPWgsMe/LhQ2xk3C07DFKi6qzjrBhsN6UHwz+e1z0mkkYj1Fw/dLmCqH7e4GahOKiTDKJ6lYIQHbs8mHgnmOMoV1jEYuZvouF66BYL6jBGEHHSNr5gpF2MoUPww26FQwfA276vjOsAaC1rAL2l2HORgweCfBQeudxa7y1oElQSxngkNN7KN8StuQpVJGtrJmN2VkXcHVvcIWlILtU6FyZqyYy5npQnX35JV4wNqhG59C2sKm+2FgVbC9FZq1STketnZWq86KPBw4pBUoxDqKDJFWmSkdglQwxvtzY5CIuSyaS6FLYy6zjiVJJdL+jNHfiGN6TTM7FgfEE33Ycsbw0Y14fCpKXgrpzJlGbzyzPI1pAGsVeDEbNwomnukqJpPWaSkYxrKK0SjEOvdpkUvqOOYDZckQpKl5oX8qfkQEBWAK0NQcRMdCSSahoziU70fZ6p/oQ1kNMAUKxfu9PX0jX84HaPJNoNT94cWO97v6/wQfCHTssWjG6tnjsHLQJRuxRDzRh88i5pOAEY9PQwK13RKtGIdp+EC2ZaC1VzWMMUjG6pqwmjuVD6u5q6AYwC4kPIdqAUv7kILzWysnZCYz3TuZZmmNWm+fvYsmOGmjxmHAhBWPBY9pd4RWHIJJGxfNOUr6gkDHxXylEeFgqS811zJQDVS07SYdHW4MWg6aewRUbHUzeMrVwRMvN/pHoeSgh45ECeOJPmSZYuhoRjw+GSar7Y1QrDYMWjtHN0DL7V4A1Xu1E0CMwLYS8JzHOpCpqXk4jLE68RyPBDq0QqVNzdMgUngc62wiCFvrDGCAyxnAAEsPE6FyjyKmeCDwdiyaM5YjxjoM5wPM8E3A2ePro8vd3usfamn3NNoNemrajmUwQq9jcUd0ltfbMm6yVFZv+5fOyqq6H2/7OwHgf7wdunzYFnBAo5Ey4EG87qUqjSf60MnGMBGNeBxmajAvAmtNqOLhGsJ0IQcA70IOwDJEr7ssnUcG/MTGNSLDx1d4CVin8qbhA4QQcnTmTc0tyDgIQa21GrXeygdyzbFQkN33kYzHc3lnNHHcjCbunNHUdlS4wmD2gZ8BxPEzgDh6BhD3mQHEfWYAcdwMII6ZAcTRM4A4ZgYQh88AoMXGDO1xn6E9jh/a43UM7fHK0O5N3OP4vSYMHZi8FfB6bY8OcQ675sHMCLR8SI3IsqLa3KoWhJYZ/cNgMmA5Q2GCeKIPmn8Mjhvx+HhveK2l1BTPYSVwmW0hNUqqWkeNfj2WMdgrt6xinDfUfjZpP8W3YpamdYmM3YzRnky+6mJcoZUpYCUUfK9fy4JQ60g5CGv/bT9gHYEmxntE8WIqtcIdeyQGaa7ILPPzVqkS2NQ8GkbGE/mY9SLJk463upXASs9xk9afdmmHoeSeiHGzLxEzHUXY1ZaXz7Ip5BLd3eA8LKMAvl6c2pgvxGfP7jNjhUxubhWdOXPm/Nmzo1HrILyYaR3oEc0nM8tBxJ658ysTsHK6WSOXB5s6WP+P5orJsplCgWFumX3qHngOXWr0it8oHdNtLrNONQF7UoxHE0afbQNoxHr6DhB0Pt2ofGI9Ztnuc6aRTGZiNYlYiHV+i9kP3hzkR3kI18/RHiHI2TpjKh9NFMqP2VzSyucLeevjlTTgvIXwcIlC+f2I5QTAqpesUSmJM6zUJtZ5JuAslj5w8oZ9GltXKmREu8101CjGE5loNpE1wSfjQG1mFpzbncoWxvtQ3WZ6AgapnlmDZKxzX8AEGEkApUQT+WihNGIhQaCkTLGAz68rWcz3AENHEnETrB2ZuTSRXmrMnt1nJItmtJiOm12JtBmPApdRE9bU3IjGCyuyZlNzExoA5QYOuJBIgxslNGguKyb6jKSZLoxyQYUiOJSmktVQt7DyxxBvuI2P8AZnsmbOKGRy8+YhRePmDfOK0pm0uaxoJGtySaT7Mr0mJDhv5go1Ze0x8tFec8UYb7hbSSO9YiMfzRdyiXR3U/NwiCxnZnNNzRMgkkwububAkQGdS81Yoam5pl5GHnwTDlETOA0vmTEK8Bw7M5kkPJKRyxkrvO1sFx4QxXRiWdGM9hjATEcjILvlxiGkVTsZjyAc6mpEI7bWMIBdSbdp24p09JNyQFPzJATo/j0RQWEUU4VsxaCSSOSt6OB0kabmMQioZIGowroNcRyCqjbBcARheBrHIak2zlQ04jXcRjRqN5O7z4DuWlXYBKjMXdWxUKZa0Uao3FGZyS6gd7mR63aYSem3tywVzNX8o+CMXZlGuLBaUnejJdK15jUCRtipj4eJcJoqIQ5FuM2zq5iOOfRg//TmU4ZcpRwJRexiwmW2Lx4LlVX1MwYqrzgEdztaPtBZfvu3tx0rGKYdK4xdhTFwYXmgmQgX4/pnBXI0xlgkYXcbdzYgzFFZ+6e3scoQprHKCKz7lWXumjRCGUdF3IlYo4yPviuMXQqEMJvJRo0CInWXR22EM1XDGgYHDE+PrApQzeQk7GZC5F0db6eXgGLayFlT+ZyZt9fCrvQGVU2vFu5L5BOFsUixrclJSLnZb8aKBXs7u3I/sqdvyUTBzBnJ2bNXun5Xm8WDub2RR+jMRi0jmaX5TLo8ohTMfMGtBU9IdYytQS0djEFJ3UNwjdhZtJYKlKttldrAaivDIlilGocB7IJNwRDOspVnNaUJT7V53AFVNXlBd9f2Sp1ZjXczSzPgiNGMfTatmTLTBTM+tIIYBSvI0wqOcGfK5buYfNZ6juDSb01YtZNBcCu3RrTcrm25k+WTiZjpyc4TVO1ktbCrk9WK3Z2sVg7rZPnSu6iSkla6fle9swezCjIKLnN3QI8Q2gj2Ej+3Vrxhjkaoxd2NUCu3izQZDcA6X84s9OQyy9NRsx+cdGqXqzaw2rg5I5E3427cE9TUXL7fy+Yy4IGQd4ypBMP6gfuxk0uzQSiSAs9Gco6aRo1cdxHwjhtqTxQjlstU050MhaxAp5nAS9iVyTkgOJJIFlwlrKrTDVqPbXKJfCbtgofA4OqtV2m6X3VS7oCqk/KCbifllcIap8TAnFQ5k3Qmk63tKeVuUNNYs2evrAmrdgMI7uoGELm7G0AAZ73KA6LHIECh3CFVJdagLldcI3UPiDViZ2HKzZw0+0Erl0/MHewOTuSjyzO5+LCaULB5XbeZW1gS9JorAFfpDG6dY6RNzc31JGFVe2o9pK2ClnpQpzrKKi3Vy2u2E1FiZxoVqMvVuM6f1fs9N2TVbyRU5H4A45Y5cy9baSJupguJrgRoqJXVH9VnR07ATnwsROJMulwwl/fxDFkemXvI8gidaVfiW7dDiDmnR+iMX/ZtJbfnVLw3qDoPqIVd84BasXseUCuHDcI1jrhaJJj3geAu7wORu70PBHAWa1AJi5tdRjFZiFq3L2WzAo//E2nroZunzs0YIp/oToP2WN4Dnk1OwZAw/YAPaEGXdDZZTVh1Ig7B7RKWnVPMyOaBI6vcyDVVBMmkOxdXQHWq4AUt/Y9GCN3jmVcK8/sW4zIHT0jV79egLr9fI3U7qRoxzNFZL4tr6zMZJXa39UQU5syqbNmdSSPdCylv2RA6E7W3ZDVh1Y4CwV0dBSJ3182+Za/OYFy/q9bgwdweySN0VltPLAtE85lcIdpjJrNmbnI3eE6Qynfbrw9z5rLZs2uCmpoHd3cXu6L2u3iwFz247zFzw61Qa196O6Cy9Hh6d3cqGV1udnZni9F8jyVLJjpLCyqy+ZiRNCvvmqZiYUtfZXQGDs0XU9FcZnm+TjqfiubBEoG66Vgm3VehZ2LpTLLPBK+j60w80wVOC+ivkzYL7mo2+dF1KdrdJtgS5DLLo+lMLlWhm/F0tppuAEum8na6qWKyEmMaNoaZNY1qBVtw7LKikS4k9jWjy9rrU13WiNeXMihuyihE+8xYJUKongiJuCvOrDrjlPnWOvluo9ADlkXU0wblaF1GvlCfmrqTxfrMpttruLNxtLXjTtQoFNKWjnJmvBgz61OvJ2pd6nXEqcvyYsUU8Dd1aSiWXVFnopl0zGHOWLY0kpRZrMc1ct2Wzy/DwXrglJnrNusri5HrdrovPBuPO214iJOt2uwIZ7C7PYdaItCw8UwhWlgWjLaALr3YGx4ohQ9xhS9rt4NbPMGt0cUwOoIKDoDgQE0wNO0wPJEwPJEwPJEQPJEgKhiaSEknLW5dpfu6smFIeKq/HD7MFZ5YFo72561sawXppBVjuEcQivaXogytkdjh3hjBSgxvJsEoIqlgFB4eQIanrPANXOGdXYHWgVZIoQfMa8DGvNG0uVyzwtJFsCItnSgo1s9E3gqZYP2wV1CBZ1dFa8UVOGYOdFI7nUEWEwMGnQ0Fo4VMNBEKDq4JBP8fZYeWEurK5JYbubg1oegKBcdaQjDNMtPxSjdbkY715DLgMclEqNwslFbbRQ3ANsKgnNlteem0kSp5NARgr8SLxjLFdGG8PzgUgQyDhYPMx8EELu1Co3blTHMyTFAqb76YzWZyhXw0k4VWz4uBz9Ch2iyB1iwhl8nmJ/hAoE6TfBh7oexUHypu5mO5hPV02Q/tLHZ1mbkouDfAVcLsA3euaXM5roA2BPQLVZwjq0p9p9cDgmEjn9i3NJ32ga2FiRY+oz480Z0Gd36+JbETtkOgWijB1U6E00LKTFVBXHLdFQpqPw4KbLECNewSAxqm9De0Y5cwa9UzHCj9tNZEuoBUMVlIQPIZi6asbEa75WbBcPqo8RCpx0NBkG4PMgKOAOsbUisChZ9YG1zrGyb7Q8AzTEJipYXplgBSDa/3GIdFQG0mYAnbc8zAMmDxttPmm7C0w8vgQUeK4VqwunAXJYLo0NkxwbqKTL4AaRCYx5nqj5X9zXR/tOptptUDQ31NDev0NNG0T4GdMMQCapzSGCRjLS9Ga9HlsSBdpIRZZg1zWE6o2kchZu1xaeisHI4GXSvLz4xyiWOlqQ44ETpndo2FCqvTujFQebpEeIb1stjoBLdH7kddTiZnZsEXBTALbakHdjZ7s0+EarNN9SEdjTcJgtbO62p156Yaa+T2BC2bNNJW+43HAfZ+PrUKdiL2TK+2jVwTwXG1YtuvxsCtJnjKOhpKdJlGoZgz84NrpKDstQWrHSZq1eg/f3RQzkGithLeMaJWmTUDwEwcYvYXcuWOk7fsbAoOdwwAWM5ht7UdocRVunQmZQ9D2QLErlE9oLbiNaZf2xAwv4ZMqOqxak3a47BqVeysXLbg6e61VlTr3pDFAk+N7GLVGrDTAbYB6ezZsWwRLGIBXR6s6rImSYWckSjk53YmM7Fe667c/vTJpjb6x/E896N1Z9waXfxfZVyJ999mHPkvM478rxmH/8uMw/97xv9dG4f/1zYO/pc1Dv6vNQ78lzUO/I81Bg/H/puMq/H+y4zBQ7b/JuNqPE/GAUgCzuHC8mmzZztGPlhRITEct2t2SQaAD80K4HGa9Ro4EW9qHtSVyYFHvvaHdtGcmTPSvYNLgWYul7Emr9lMOm9OLIdWvwAsi6IZI2F/6cd1pQrRruxwM7W8O1vcNr1rJte7SSZtbpwBC7TAEtKyZOd0zMiCKUB8EchmTDl8B3NZ0cwXNrEGr0q0sR7xAvtTwoq8kuHWRnYBuDetSEaWJXaKW2XyhYpsmC3b2JoM7Wg9Kd86Ey8mzQFOgR3RFbTQ0vNIM2UP02YavOcGn7GC7yzB0c/5UTUy+7WFtaPcoLhZeR0G3uZZT9FrAwPRloGeQDAa1IS1RhfXhEWgYQFIWG16YUjcMCRuGBI3BIkbhIbVxg3UKsF6Nu4NtB6MD/YE2k/FYaHp5JCaUOt5uDdd62F4LWs9Ca9NGDwGr00hGIUEBuCBqbFxM2Wku8HGr9FiGnzYGyuY8WiPkY4nwbKsMQ55wcylEmnwKXFFvEExDZZemPHGZCbdPX/+5ObGmG3aYPGptWyv9CmqtfYFLLzfCwG4P9auQj4fa7vByWAdTiYdzWfNWDFpFBJ9JnjoVOiJ5oyCabkL8GZ+AgSrrm2wty4dimC2hoWnssloGizkjeatdaClxUJ+WFPztPpSi+eMrsJmeDaViftmm8rEm5qb6kjHyrAesNPsTqSb6wCNGFgTvdiHNLLRXv9aAKqpeWpdaVk1qQ+16uLTIiXUro1PsqVNJawSbF4XWgdUX55WVWagUKtEFeMCt8pmPo+suJu2fm1ZH1sPVWe2VoWQdbdZawvuUm2QtutA8UbuAPFG7gBts5iFJ8unTVUW9Pu0U5kv1azO1MubjvkpuMzX065lth6qzmwt1U6vj7W166Ot8klN9Vl1mbZ+1claRQ4hWLTzamoej4xTaayKJgxg9rnS/6PgKBe79lFwXgX48iCRSY/3hwd5kFgykzaHlwJzZipTsB9WWE/j04XRFYmRz6TBoN9ZjNvrrPJmYRRKCh51IIXgUQcyXas8SKm1dcoYpNSq45CSuOqerdYa6gq2N2wCe6gMg4RbC11GlgRgA9XyKuTyI95yY4P5i700udIHqkFzq/MXMBWZWlcc8NHm/IGO8MqGEKUw77RkRCkcNFguY61ZKGRy1krN8qwGHCZi3WTlo6U5V/XuaFwtY39vUyFGuAhr1gVmTqAc4+Aie4Mba+lpxEmAHWNShr0rR7Vb1IiarDkjIlZvIh1vah4HAex11GUiACFy1jEzcEFTcxtEYH/TA49U3l8AVpZyPLss4yFE6dOMMhKCIGa6mIJnDSRNzWORcew0p0PkYMWqmSuAjKPWMuTy1kjT6oLBPXqma6oPW5pUA7XO9EFLuQO7BLso+aUMvquy2znfHOsxcvPnN8Yy6Xxh8nSIydiiqc0wG7REcOVaoqbmSRCRtRuYWTlCK2d2wdK2FlvD0y59cg2z71Isu+lgtmikV0QzCAO2ZYh4Vpsh4lmy6l2HJcua3WDPuAR4M5oKAQ+aBXW2fI6RLwRq2BIAFuoaSXA3CXyRkUibOdupjUTFSBnZRpSsnN0UL1A6s9nLtXi5SmNlVxR6wIhXWAE2Ayo/a8p3ICPYrWv5Sdsjgp0erPvThLUDSugfxEwaBRAl766GfXaO/QIuZ6TzwG/aFZuA4pYtN9Mh6/Su3HgUkwIvOfNZI9c7EYt0JxOplJlD5gVaP1/IGcloCJ1XxUIacUgsm4pMRgHgxPul2WiPkUtl0itGI7FEujuELGyyKxW0Hxzkx2CZyFiUuDeRSkR7Qz7yIFIXvcFoTyaX2DeTRpayVMdofyqJzKY7W4hm8nlkGt2JbsMK7AtNQjJ270ia3UZsRROKKhs0WIrYF4oGkcmVj2G0KGT9q2f2uWwhle+umU1M8ADReKKrq/oAGvzKjwRnxkUTKaPbBKsbwUdgeTOXsHyLLbPf967IFAvFvvJBdk3Nw50yI185LRwMlVXJ8p5EPgu+PEvHKvIxDnlfpt8y/crRyU3NzkzB9m4hR6YTnLJMNmekM+Wt0qx5fSFXTI2EMaX4I5yyRHf1tDwwTDhEZjpvpjN9RrQYgJbL8g+IclmyQiEfzWd7ndWeCmO6TXAMHii+9VcJbcaiweVGX7nW+ERtElJBNwpRjgUYyLoHHXWf6JBlM7FeswCpPAIyjerxcGByB4OcRXSWI5uwTKciG+WUGTmj13QkPdoljMeTpnUoa0k63iFNg9uyHLhFCEYdtRwHQzKpdAJaxerYYG/sB20E5+gA7TypTGciaabNQl8E2nmqowJUCaURoQ+qvVQilalUL5pKjUDJvFmmMp4qOR1BMmn0GVCDsnaADEOLYo04jvxc0bpSTtEkj8hwWIe1jeJyo8/d2F7KnTwYaxzJj/SIeoMRqGbB/UQu7YjozLGnmF5RNJxSp3l154w02BI2nzXNWE8FmVCLhK3bgEwaqujuZCoMb1h7TOqDltuWFbFCeKe3hQa0wma/kUk7zpZ3xwRnlTv72liPzDo5PWrA5aUB0Rm/ES4PQs21ckx1H7TksUwavNR0dL8RLqnjzNim5kFVUbXb7AhujEr3PvYrlE4jnwDfS3Wb/XOBcLoVXH6PYg/Qji1CwGududVEADcwZqRjZhLcfyW6EjEDzB0GlMKMmPXcKlboz6vWS9zliUIP+DnY+lXoMdPRLjCh7IkWjHzvcCu0FNcWgqUvsUL/yJKkdFaNS2aA5yKN6WSmJ2Wk07NnW1NsozMB5iWBYLQFDD4FI5EsbwQyt4ra9cyDr5Gsmto9xUh6kOlWDtb3q/YePqXXS/ZXsXMr77Xy/wRO/hM48U/gWB1wtfHrgUvp9hgAnugD92RyhflNeAi8AbRfA47Hg4l0wS9Da0uF+ZPwUDxT7Eya8yfgKesJ21SjWMhU9oyx7irT9g6u4Nmr9adZ2sS5xYk6/46bBbDhSzFvb64APpAzcr1mLuITwcjnE/mCYW8OUIk1ERXL7tf2L3fSaSO5Yl/7ljZfWm4Rq87tS+nmm5rn1BGrsuWIvU9cJW7bP4mbN8Esp5DJNTXP/ifxwOPiapaB+qLmqzFmYGI4f9o3/nP/CR0tLTFJlx7HbvjfRLa8Vdp6vl9PfAduTRuTSbNU9KbmECx+5Uk42B7AG9bU3OETx95ayYwWehLpXvA83UyDfQvjTc3tdcasPorPxzJg4wGoyWEjguHHTMfMpuZZsLil9xGOCpZCmpqnY3lPwFgjm3Xt0OX63dQ8KOpYJVNI2je4Q12B1kpwK3wUJNwascCW6rBIZn+iMAISHssZ+R4zPjxas0QnZ+YLmZw5tFZib8ruDM8VgY1Gra6fKKftSjPWY4IFZ0Yi2ZnpHxCNLjfyqdKzGmvB8AalIHskLmRyeTUazRfi5WMDlPIvMIOp/ADZVLhYMpM3h0aj2VLV8mayqzLmDou66xHPFAtWWuOj0Vh/v9GZ6AuATl5qDms5a7qYtXzaFCcSN7NmOm5vkeSBQ04uGu1LJaKxpJHPl74fSXdlwJtW67jZThMsG4nnC/8gjtGZ6QMHiRTm1xEHbJlcTBupzkR3MVPMR7PFzmQiZi0cDrqj5+soZf1RKoWc5x8FU8ap7tjZjNWOzshgu4wY2FZ/phvtKqbjBvDtRhKKz3LjvlWvk6/Ue7YPj6l0k0/USh1GWWC0O5npNJKlnVXj4N1bR6g9ghS2tbcF0cJwezta2BJBC1s7WgNoYQSTZ2u4PYQWhjowyQaDGGEAV9pAuAMpjLS14YQBdJ6R1ha04iOR9jBSGG7FaCjcGkHnGWqLoGOGIiF0zGBHa2AkUhhqQUcMtqBTDbS1hkYjha1t7UGMtCXShk440tKGjhpqD2ISDrW2tmKkkbYwRhpu60BLg60htPUGgiFc1EAkgpG2BDDVDbYEAjgpTlXBlpYWtDTQ0Y6JG+jAKSPQ0YapUaCjDVPmQEcrNl+crgIdEWyNwhiLDHSEsCkHsfVtwcVt78DVqB2rq3asrtpx5hxoj2DMOdAexJYqgC0VzuoC7TirC7R14ErV1oFrwbZ2XH3bcN030NaKcQyBNqxdtUVw9W3D9e5AG9au2oI4m2wLtGOl2Pq24Mrc2o4rc2s7LuXWdlwbtbZhpa04bbRGcNpoDWPLHMamHMJqI4iNG8TGbcGVKoLt+5F2XH0jWB8bwfqNUBvOcoK4ftSC9aIt7SFMyi2tmKlYoAXXgC1hXFdoCXXgihzEdbKWlnBwDEI6swNMk9vHIsUdYKrXPgItR0dtj0ZaW9vQUTGptqJFYXSGIVBWTNQQWhQcjhQhI7WjK9DeiixleyTaEQ6iS9keaUSKQsCyg5hs0TVsD6JFyOnvzPaWCEaGVhq6iG1ofbahbaUtgtRnWwjoswWptLYQmFCHMNmildYWQItakJVvQ5a1Fcx2W9EW3NoR7Qhjelsrure1tkeDwWALOiq6j7e2gVzRysf04Va0HbdG0KIwWoRujFa0Bbei26kV3U7oPhoB7jgUwMtbMHKrHdFViaDbMdIebQ0G2pDmHGkH5tzagQWCrUG0iiPt6LhtVuLo9o6gbTvSGg12YDpMpBWdbcTKFm0wEXT3j4RBtujuHwlbaYewQLA1gjbkSBgdN2TFxegak2/QKhhGYUF03ID1kABjBAGrYGh3EMHki+4x6GYIA4tvQfuYcEc0GAygh7ZwByhwGN0KYXRlw5bFh9GtEMaUC/i+FnR3C7dZaaNdTxgTt9WK24IcQcMY9xlGd5dwxHregu7GFhDoQLd+GNMQVncIYQqGHr7CYChuQXfjMNojhoPoVAMgVUzrov1wuCXaFu5owckjrQGMdbSAp4OYFm4Bim5Hzy3C6LxD1pQfPQaFrEk/2t+GrB6H1mioA2l3oTb0rC6EceEh9MAQQrd6CN10oRC6esFoazCEKQu6S2IG7lALSBXdMUJo34cuahDcQ3W0I1Ua7EBPsIOYXhpEt2CwHSfDFKW9DSNDG2OwDVQRbYzBNvAWIoROuw0nC2JkgXFoWUs0EAi3YJTbhlEupjKt4A0GerwJtoJn5S1hdL6tHehSt7aDm7gOTPu1om8AMP4iGAFN1IpONoJpAsxMIYge9IPhaKStHT1NCYaBj8MM6kH0WBIMgeqg/VQwhJNhqoPpxwGQJaZIAVBdtHMJBsBbKLRHDgaQmgp0WAMJ2r8G0HcMgXZQLPQMONBupY0e8gHQEsbcIQfQU41AmxUXU3B0PwtYFotRSsR6Y4XuKYEIug8GMDPcANp0AiG0gwyEMNlhhqoA+iYgEAQv1trRU8RAEO3MA0GM2tHDYwDtMAOYxz2BFrT/CGAeBQWGIiQDQLgRzRZzZrQvkSsUjaRiBxkFsEBnUnSzrbZduGCraDRf7IxuYX+qAj5MiZbWEuVmxrLZcbtutt3O9j48W9ifebsDmpqbvcSuiWRyk1wmu5WRLyzqt1fE7AA+PbVIewMfH5IgSIIiaIIhWIIjWIInBEIkWIIlJIIlZEIhVEIjdCukgWCJDYgBxEBikPWbJQYTLDGEGEoMI4YTI4iRxCiCJUYTLDGGGEs0EuOI8cQEYiIxiZhMTCGaiGaCJaYS04jpxAxiJjGLaCECRLCUUogIExGilWgr/W4nOojZxByCJeYS84j5xIbERsQCYiHBEhsTmxCLiE2JzYjNiS2ILQmWWExsRWxNbENsS2xHbE/sQOxI7ETsTOxC7ErsRuxupbYHwRJ7EnsRYwiWWELsbf0/SuxDbEQYRCcRI+KlXFnCJLqIbqKHSBAssZRgiV6CJZJEikgTGSJLLCNyRJ4oWGSR6COWE/3ECmJfYj9i/1L8A4iVxIHEQcQY4mDiEOJQ4jDicOII4kjiKOJo4hjiWOI4giWOJ1ZV8juBOPE/Wj2JOJk4hTiVOO0/+jydOIM4syI/izibYIlziHOtX+cR5xMscQFxIXERESYuJi4hLiVY4jLicuIKgiWuJFjiqv/o+GriGmI1sYa4lriOuJ64gbiRuIm4mbiFuJW4zUrlduIO4k7iLuJu69c9xL3EfdZf9xMPlHJ9kHiIeJh4hHjU+vUY8TjxBPHkfzT9lPX7aeIZ4lniuUoZnydeIF4kXiJeJl4hXiWWEa8RrxNvlGRvEmuJy4i3iLeJd/6j83eJ94j3iQ8syYfER8THxCfEp9avz4jPiS+IL4mvCJb4mviG+Jb4jlhnSb4nfiB+JH4ifibmE78QvxIywRK/Eb8TfxB//kfffxF/E+sJgiRJimQJmmRIluRInhRIkZRImWSJ6qWQKqmROtlAbkAOIAeSg0rSweQQcig5jBxOjiAHkiPJUeTokmQMOZZsJMdZv8aTE8iJ5CRyMjmlJG0im8mppEZMI6eTMxw5zSRnkS1kgAySITJMRshWss2StpMd5GxyNjmHnEvOI+eTG5IbkQvIhY6YG5ObkIvITcnNyM3JLcgtycXkVuTW5DbktuR2Dmp7cgdyR3IncmdyF3JXcjdyd3IfYg9yT3INsRe5hNybjJL7kAbZScZctY+TcdIku8husodMkEvJXjJJpsg0mXFR4MqSy8gcmScLZJHsI5eT/eQKcl9yP3J/8oAaFlwryQPJg8iDyUPIQ8nDyMPJI8gjyaPIo8ljyGPJ48jjyVXkCeT69Se64p5ELiNOJk8hTyVPI08nzyDPJM8izybPIc8lNyLOI88nLyAvJC/y5HYxeQl5KXkZeTl5BXkleRV5NXkmcQ252tPKLLGGvJa8jryevIG8kbyJvJm8hbyVvI28nbyDvJOcT9xF3k3eQ95L3kfeTz4ArQ9LPEg+RD5MPkI+Sj5GPk4+QT5JPkU+TT5DPktuSTxHXk1eTT5PvkCeSbxIvkS+7EjjFfJV8l1iDPka+Tr5BvkmuZZ8i3ybfId8l3yPfJ/8gPyQ/Ij8mPyE/JT8jPyc/IL8kvyKvJP8mpxPfEN+S35HriNXk9+TPyBK5b5+JH8ifyZ/IX8lfyN/J/8g/yRvIf4i/ybXkwRFUhRFUwzFUk8QHMVTAiVSq0mJkil8mgqlUhqlUw3UBtQAaiA1iBpMDaGGUsOo4dQIaiQ1ippEjqbGUGOpRmocNZ6aQE2k1hCTqMnUFKqJaqamUtOo6ZRMzqBm+uQErllUCxWgVGoTMkiFqGVEmIpQrVQb1U51ULOpOdRcah41n9qQ2ohaQC2kNqY2oRZRm1KbUZtTW1BbUoupraitKY3YhtqW2o7antqB2pHaidqZ2oXaldqN2p3ag1pN7llHKarXXtQSam8qSu1DGVQnFaPilEl1Ud1UD5WgllK9VJJKUWkqQ2WpZVSOylMFqkj1UcupfmoFtS+1H7U/dQC1kjqQOog6mDqEOpQ6jDqcOoI6kjqKOpo6hjqWOo46nlpFnUCdSJ1EnUSdTJ1CnUqdRp1OnUGdSZ1Fnf2Pygq7zqHOpc6jzqcuoC6kLqIupi6hLqVeJy+jLqeuoK6krqKupq6hVlNrqGup66jrqRuoG6mbqJupW6hbqduo26k7qDupu6i7qXuoe6n7qPupB6gHqU+Jh6iHqUeoRymFWEMAu3+Mepx6gnqSeop6mnqGWk0+Sz1HPU+9UHfZvydepF6iXqZeoV6lXqNep96g3qTWUm9Rb1PvUO9S71HvUx9QH1IfUR9Tn1CfUp9Rn1NfUF9SX1FfU99Q31LfUeuo76kfqB+pn6ifqV+oX6nfqN+pP6g/qb+ov6n1FEGTNEXTNEOzNEfztECLtETLtEKrtEbr9GpytdXDGugNaJkcQA+kB9H/q9ad12B6CD2UHkYPp0fQI+lR9Gh6DD2WbqTH0SQxnh5PT6An0hsTk+jJ9BS6iW6mp9LT6On0DHomPYtuoQN0kA7RYTpCt9JtdDvdQc+m59Bz6Xn0fHpDeiN6Ab2Q3pjehF5Eb0pvRm9Ob0FvSS+mt6K3preht6W3o7end6B3pHeid6Z3oXeld6N3pxdTe9B70nvRS+i96Si9D23QnXSMjtMm3UV30z10gl5K99JJOkWn6TVEhs7Sy+gcnacLdJHus7S1nO6nV9D70vvR+9MH0Cv/j2oLfR1IH0QfTB9CH0ofRh9OH0EfSR9FH00fQx9LH0cfT6+iT6BPpE+iT6ZPoU+lT6NPp5cRZ9Bn0mfRZ9Pn0OfS59Hn0xfQF9IX0RfTl9CX0pfRl9NX0FfSV9FX09fQq+k19LX0dfT19A30jfRN9M30LfSt9G307fQd9CriTnoQdRd9N30PfS99H30//QD9IP0Q/TD9CP0o/Rj9OP0E/ST9FP00/Qz9LP0c/Tz9Av0i/RL9Mv0K/Sr9Gv06/Qb9Jr2Wfot+m36Hfpd+j36f/oD+kP6I/pj+hP6U7iU/oz+nv6C/pFn6K/pr+hv6W/o7eh39PT2Z+oH+kf6J/pn+hf6V/o3+nf6DXr/+T/ov+m96PU0wJEMxNMMwLPMsxTE8I5MCIzISIzMKozIaozMNzAbM/zut8/+9awAzkBnEDGaGMEOZYcxwZgQzkhnFjGbGMGOZRmYcM56ZwExkJjGTmSlME9PMTGWmMdOZGcxaaiYzi2lhAkyQCTFhJsK0Mm1MO9PBzGbmMHOZecx8ZkNmI2YBs5DZmNmEWcRsymzGbM5swWzJLGbWUFsxWzPbMNsy2zHbMzswOzI7MTszuzC7MrsxuzN7MHsyezFLmL2ZKLMPYzCdTIyJMybTxXQzPcxOZIJZyvQySSbFpJkMk2WWMTkmzxSYItPHLGf6mRXMvsx+zP7MAcxK5kDmIOZg5hDmUOYw5nDmCOZI5ijmaOYY5ljmOOZ4ZhVzAnMicxJzMnMKcypzGnM6cwZzJnMWczZzDnMucx5zPnMBcyFzEXMxcwlzKXMZcznzKTGGWk9dwVzJXMVczYSJa5jVzCpiDXMtcx1zPXMDcyNzE3MzcwtzK3MrcRtzO7OGuIO5k7mLuZu5h7mXuY+5n3mAmUU8yDzEPMw8zDzCNNCPMkOJx5jHmSeYJ5mnmKeZZ5hnmeeY55kXmBeZl5iXmVeYV5nXmNeZN5g3mbXMW8zbzDuM7a1Xk48y7zLvMquo9xgwdr7PfMB8yHzIfF/xTUcTPPMRI5MfM58wnzKfMZ8zXzBfMl8xXzPfMN8y3zHrmO+ZHxid0ZkfmZ+Yn5lf/n/bT35lfmN+Z/5g/mT+Yv5mXiHXMwRLshRLswzLshzLswIrshIrsxdRCquyGquzDewG7AB2IDuIHcwOYYeyw9jh7Ah2JDuKHc2OYceyjew4djw7gZ3ITmIns1PYJraZncpOY6ezM9iZ7Cy2hQ2wATbIhtgQG2YjbCvbxrazHexsdg47l53Hzmc3ZDdiF7AL2Y3ZTdhF7KbsZuzm7Bbsluxidit2a3Ybdlt2O3Z7dgd2R3Yndmd2F3ZXdjd2d3YPdk92L3YJu4zYm42y+7AGu5rqZGNsnDXZLrab7WET7FK2l02yKTbNZtgsu4zNsXm2wBbZPnY528+uYPdl92P3Zw9gV7IHsgexB7OHsIeyh7GHsYezR7BHskexR7PHsMeyx7HHs6vYE9gT2ZPYk9lT2FPZ09jT2TPYM9mz2LPZc9hz2fPY89kL2AvZi9iL2UvYS9nL2MvZK9gr2avYq9lr2NXs3fQa9lr2OvYw5nr2BvZG9ib2ZvYW9lb2NvZ29g72TvYu9m72HvZe9j72fvYB9kH2IfZh9hH2UfYx9nH2CfZJ9in2afYZ9ln2OfZ59gX2RfYl9mX2FfZV9jX2dXYg+Qb7JruWfYt9m32HfZd9j32f/YD9kP2I/Zj9hP2U/Yz9nP2C/ZL9iv2a/Yb9lv2OvZFYx37P/sD+yP7E/sz+wv7K/sb+zv7B/sn+xf7NrmcJjuQojqdojuFYjuN4TuBETuJkTuFUTuN0roHbgBvADeQGcYO5IdxQTiEUYhg3nBvBjeRGcaO5MdxYrpEbx43nJnATuUncGmINMZmbwjVxzdxUbho3nZvBzeRmcbOIFi7ABbkQF+auISPcZCpDt3JtXDvXwc3m5nBzuXncfG5DbiNuAbeQ25jbhFvEbcptxm3ObcFtyS3mtuK25rbhtuW247bnnmeeZ3bg1q9fv35HbiduZ24XblduN253bg9uT24vbgm3Nxfl9uEMrpOLcXHO5Lq4C8hu618318O9ySQ4cC3l3mZI6+rlVpNJLslV/UyKK1+rqPeZ95g0l+FOoCZR35MfMCdSKQ54qSy3jFvG5bgPmfK1iqKZBjrPMaUYLGPHeZQ5keKYAlfk+qxrOdfPreD25VbQ+3H7cftz+3MHcCu5A7mDuIO5Q7hDuUO5w7jDuSO4I7mjuKO5Y7hjueO447lV3AncidxJ3MncKdyp3Gnc6dwZ3JncmdxZ3NncOdw53Lnc/22f8+/17/V/6jqPO5+7gLuQu4i7mLuEu5S7jLucu4K7kruKu5q7hlvNvUKu4a7lruOu527gbuRu4m7mbuFu5W7jbufu4O7k7uLu5u7h7uXu4+7j7uce4B7kHuIe5h7hHuUe4x7nnuCe5J7inuYmMs9wz3LPcc9zL3Avci9xL3OvcK9yr3Gvc29wb3Jrube4t7l3uHe597j3uQ+4D7mPuI+5T7hPuc+4z7kvuC+5r7ivuW+4b7nvuHXcOu577gfuR+4n7mfuF+5X7jfud+4P7k/uL+5vbj1H8CRP8TTP8CzP8TzP8QIv8lLpvzKv8Cqv8Tqv8w38BvwAfiDfxg7iB/ND+A52KD+MH86P4Efyo/jR/CpiDD+Wb+TH8eP5CfxEfhI/mZ/CN/HN/FR+Gj+dn8HP5GfxLXyAD/IhPsxH+Fa+jW/nO/jZ/Bx+Lj+Pn89vyG/EL+AX8hvzm/CL+E35zfjN+S34LfnF/Fb81vw2/Lb8dvz2/A78jvxO/M78Lvyu/G787vwe/J78XvwSfm8+yu/DG3wnH+PjvMl38d18D5/gTyKW8r38AibJp/g0n+GbySx/CLGMz/F5fneiwBf5Pn4538+v4Pfl9+P35w/gV/IH8gfxae5g/hD+UP4w/nD+CP5I/ij+aP4Y/lj+OP54fhV/An8ifxJ/Mn8Kfyp/Gn86fwZ/Jn8Wfza/I3MOfw5/Ln8efz5/AX8hfxF/MX8Jfyl/GX85fwV/JX8VfzV/Db+aX8Nfy1/LX8dfz9/A38jfxN/M38Lfyt/G387fwd/J38Xfzd/D38vfx9/PP8A/yD/EP8w/wj/CP8o/xj/OP8HvzjzJP8U/zT/DP8s/xz/Pv8C/yH9BvMS/zL/Cv8q/xr/Ov8G/ya/l3+Lf5t+xrnf59/j3+Q/4D/mP+I/5T/hP+c/4z/kv+C/5r/iv+W/4q+hv+e/4dfz3/A/8j/xs6ie+hzmC/Zn/hf+V/43/nf+D/5P/i/+bX88TAilQAi0wAitwAi8IgihIgiwogipogi40CBsIA4SBwiBhsDBEGCoME4YLI4SRwihhtDBGGCs0CuOE8cIEYaIwSZgsTBGahGZhqjBNmC7MEGYKs4QWISAEhZAQFiJCq9AmtAsdwmxhjjBXmCvME+YLGwobCQuEhcLGwibCImFTYTNhc2ELYUthsbCVsLUA7p+3EbYVrma3E7YXdhB2FA5mdhJ2FnYRdhV2E3YX9hD2FPYSlgh7C1FhH8EQOoWY8CgTF0xhPQmuLqFb6BESwlKhV0gKKSEtZISMkBWywjJhmZATckJeyAsFoSAUhaLQJ/QJy4XlQr/QL6wQVgj7CvsK+wn7CwcIK4UDhYOEg4VDhEOFw4TDhSOEI4WjhKOFY4RjheOE44VVwgnCicJJwsnCKcKpwmnC6cIZwpnCWcLZwjnCucJ5wvnCBcKFwkXCxcIlwqXCZcLlwhXClcJVwtXCNcJqYY1wrXCdcL1wg3CjcJNws3CLcKtwm3C7cIdwp3CXcLdwj3CvcJ9wv/CA8KDwkPCw8IjwujVPe1R4THhceEJ4UnhKeFr4iXpGeFZ4TnheeEF4UXhJeFl4RXhVeE14XXhDeFNYK7wlvC28I7wrvCe8L3wgfCh8JHwsfCJ8KnwmfC58IXxPfk9+KXwlfC18I3wrrGOX0Eto8NTuRuI7dgkN/t1IrKXX0kvo74R1wvfCD8KPwk/Cz8JP7C/Cr8Jvwu/CH8Kfwl/C38J6gRBJkRJpkRFZkRN5URBFURJlURFVURN1sUHcQBwgDhQHiYPFIeJH9FBxmDhcHCGOFEeJo8Ux4liRp7ajGsVvyDAxThwvThBXERNFmZskThaniE1iszhVXEWsIqaJ08UZ4kxxltgiBsSgGBLDos5FxFaxTWwXO8TZ4hxxrjhPnC9uKG4kLhAXivdTQ7iNxU3EReIiUSE2FTcTNxe3ELcUF4tbiVuL95DbiNuK24nbizuIO4o7iTuLu4i7iruJa4jdxT3EPcW9xCXi3mJU3Ec0xE4xJsZFU+wSu8UeMSEuFXvFpJgS02JGzIrLxFzlAvO19xnvbA3M8HIieOqfFycTEbIgFsU+cbnYL64Q9xX3E/cX53AHiCvFA8VVxEHiweIh4qHiYeLh4hHikeJR4tHiMeKx4nHi8eIq8QTxRPEk8WTxFPFU8TTxdPEM8UzxLPFs8RzxXPE88XzxAvFCEcxlLxIvFsFs9hLxUvEy8XLxCvFK8SrxavEacbW4RrxWvE68XrxBvFG8Sbxb6OROI24Wr+VvEfu4W8VbxdvE28TbxdvFO8Q7xDvFO8W7xLvEu8W7xXvEe8R7xfvE+8UHxAfFh8SHxUfER8XHxMfFJ8QnxafEp8VnxGfF58TnxRfEF8WXxJfFV8RXxdfE/9vj/r/Xv9e/17/Xv9e/17/Xv9e/17/Xv9e/17/Xv9e/17/Xv9e/F+56XXxDfFNcK74lvi1eyb0jviu+J74vfiB+KH4kfix+In4qfip+Jn4ufiF+IX4pfiV+JX4tfiN+K34nrhPXid+LP4g/iD+KP4k/iT+Lv4i/iL+Kv4m/ib+Lf4h/iH+Kf4l/i+vF9SIhkRIpUdJ1HC0x0vUcK3ESLwnSjZwoSZIsKZIqaZIuNUgbSAOkgdIgiWUHS0OkodIwabg0QhopjZRGSaOlMdJYqVEaJ42XJkgTpUnSZGmK1CQdRDZLU6Vp0nRphjRTmiW1SAEpKIWksBSRWqU2qV3qkGZLc6S50lxpnjSKni9tKG0kLZAWShtLm0iLpE2lzaTNJZ3dQtpSWixtJW0tbSNtK33KbSdtL11C7SDtKO0k7SztIu0q7SbtLu0h7SntJS2R9pai0j6SIXVKMSkumVKX1C31SAlpqdQrJaWUlJYyUlZaJuWkvFSQfueKUp+0XOqXVkj7SvtJ+0n7SwdIK6WV0oHSgdJB0kHSwdIh0qHSYdLh0hHSkdJR0tHSMdKx0nHS8dIq6QTpROkk6WTpFOlU6TTpdOkM6UzpLOls6RzpXOk86XzpAulC6SLpYukS6VLpMuly6QrpSukq6WrpGmm1tEa6VrpOul66QbpRuknamL1ZukW6VbpNul26Q7pTCvN3SXdL90it/L1SG3+fdL90v/SA9KD0oPSQ9LD0sPSI9Kj0mPSY9Lj0hPSk9JT0lPS09Iz0rPSs9Jz0vPS89IL0ovSS9LL0svSK9Kr0mvSa9Lr0hjSHf1NaK70lvS29I70rvSe9L30gfSh9JH0sfSJ9Kn0mfS59IX0pfSV9LX0jfSt9J62Tvpd+kH6UfpJ+ln6RfpV+k36X/pD+lP6S/pbWS4RMypRMy4zMypzMy4IsypIsy4qsypqsyw3yBvIAeaA8SB4sD5GHysPk4fIIeaQ8Sh4tj5HHymPlRnmcPE7u4cfL4+UJ8kR5opzkJ8mT5cnyFLlJbpab5anyNHm6PEOeKc+UZ8ktckAOyiE5LEfkVrlNbpPb5Q55tjxHnivPlefJ8+UN5Q3ljeQF8kJ5obyxvIm8SF4kbypvJm8ubyFvKS+Wt5K3lreRt5W3k7eXd5B3kHeUd5J3lneWd5F3lXeTd5f3kPeQ95T3kpfIS9i95ai8j2zInXKnHJPjcpo35S65S+6We+SEnJCXyr1yUk7KKTktZ+SMnJWXyTk5L29GFOSi3Ccvl/vlFfK+8n7y/vIB8kr5QPkg+WD5EPlQ+TD5cPkI+Uj5KPlo+Rj5WPk4+Xh5lXyCfKJ8knyyfIp8qnyafLp8hnymfJZ8tnyOfK58nny+fIF8oXyRfLF8iXypfJl8uXyFfKV8lXy1fI28Wl4jXytfJ18v3yDfKN8k3yzfIt8q3ybfLt8h3ynfJd8t3yPfK98n3y8/ID8oPyQ/LD8iPyo/Jj8uPyE/KT8lPy0/Iz8rPyc/L78gvyi/JL8svyK/Kr8mvy6/Ib8pr5Xfkt+W35Hfld+T35c/kD+UP5I/lj+RP5U/kz+Xv5C/lL+Sv5a/kb+Vv5PXyd/LP8g/yj/JP8u/yL/Kv8m/y3/If8p/yX/L62VCIRVKoRVGYRVO4RVBERVJkRVFURVNOZPXlQZlA2WAMlAZpAxWhihDlWHKcGWEMlIZpYxWxihjlR2ZRmWcMl6ZoExUJimTlSlKk9KsXMhPVaYp05UZykxlltKiBJSgElLCSkRpVdqUdqVDma3MUeYq85T5yobKRsoCZaGysbKJskjZVNlM2VzZQtlSWaxspWytbKNsq2ynbK/soOyo7KTsrOyi7Krspuyu7KHsqeylLFH2VqLKPoqhdCoxJa6YSpfSrfQoCWWp0qsklZSSVjJKVlmm5JS8UlCKSp+yXOlXVij7Kvsp+ysHKCuVA5WDlIOVQ5RDlcOUw5UjlCOVo5SjlWOUY5XjlOOU45VVygnKicpJysnKKcqpymnK6coZypnKWcrZyjnKucp5yvnKBcqFykXKxcpV/CXKpcplyuXKFcqVylXK1co1ympljXKtcp1yvXKDcqNyk3Kzcotyq3Kbcrtyh3Kncpdyt3KPcq9yn3K/8oDyoPKQ8rDyiPKo8pjyuPKE8qTylPK08ozyrPKc8rzygvKi8pLysvKK8qrymvK68obyprJWeUt5W3lHeVd5T3lf+UD5UPlI+Vj5RPlU+Uz5XPlC+VL5Svla+Ub5VvlOWad8r/yg/Kjcwv+k/Kz8ovyq/Kb8rvyh/Kn8pfytrFcIlVQplVYZlVU5lVN5VVBFVVJlVVFVVVN1tUHdQB2gDlQHqYPVIepQdZg6XB2hjlRHqcvY0eoYdazaqI5Tx6sT1InqJHWyOkVtUpvVqeo0dbo6Q52pzlJb1IAaVENqWI2orWqb2q52qLPVOepcdZ46X91Q3UhdoC5UN1Y3URepm6qbqZurW6hbqovVrdSt1W3UbdXt1O3VHdQd1Z3UndVd1F3V3dTd1T3UPdW91CXq3mpU3Uc11E41psZVU+1S7+O71R41oS5Ve9WkmlLTakbNqsvUnJpXH+ULalHtU5er/eoKdV91P3V/9QB1pXqgepB6sHqIeqh6mHq4eoR6pHqUerR6jHqsepx6vLpKPUE9UT1JPVk9RT1VPU09XT1DPVM9Sz1L3Z05Wz1HPVc9Tz1fvUC9UL1IvVi9RL1UvUy9XL1CvVK9Sr1avUZdra5Rr1WvU69Xn+RvUG9Ub1JvVm9Rb1VvU29X71DvVO9S71bvUe9V71PvVx9QH1QfUh9WH1EfVR9TH1efUJ9Un1KfVp9Rn1WfUz+knldfUF9UX1JfUl9WX1FfVV/mX1NfV99Q31TXqm+pb6vvqO+q76nvqx+oH6ofqR+rn6ifqJ9a12fqZ+rn6hfql+pX6iv81+o36jfqt+pa/jt1nfq9+oP6o/qT+rP6i/qr+pv6u/qHGmXe5f9U/1L/VterhEZqpEZptHUxGqtxGq/xmqCJmqTJmqKpmqbpWoO2gTZAG6gN0gZrQ7Sh2jBtmPYl/yU/XBuujdBGaqO00dq3/BhtrDZWa9TGaeO17/gJ2kTrmqRN1tbxU7QmrVmbqk3TpmsztJnaLK1FC2hBLaSFtbAWsa5WrVVr09q1Dm22Nkebq83VfuB/4Odp87T52obaRtoCbaG2sbaJtkjbVNtM21zbQttSW6xtpf3Ib61to22r/cRvp22v7aDtqO2k7aztou2q7abtru2h7antpS3R9tai2j6aoRlap9apxbS4Zmpd2t98t9at9WgJLaEt1Xo1UkhqKW0endYyWlZbpuW0vFbQilqftlzr11Zo+2r7aQFhfy0iHKCt1A7UDtIO1g7RDtUO0w7XDteO0I7UjtKO1o7RjtWO047XVmknaCdqJ2kna2C9xCnaqdpp2unaGdqZ2lna2do52jzhXO087XztAu1C7ULtIu1i7RLtUu1p4jLtcu0K7crSdZV2tXaNtlpbo12rXaddr12v3aDdqN2k3azdot2q3abdrt2u3aHdqd2l3a3do92r3avdp92vPaA9qD2kPaw9oj2iPao9pj2uPaE9qT2lPa09oz1OPKs9Z13Pay9oL2ovaS9pL2uvaK9qr2mvaa9rb2hvamu1tdpb2tvaO9q72rvae9r72gfah9qH2kfax9on2qfaZ9rn2ufaF9qX2lfaV9rX2jfat9p32nfaOu177QftR+1H7SftZ+0X7VftV+037TdtU+F37Q/tT+0v7W9tvUbopE7pNFiErbM6p/O6oIu6qEu6rCu6qqu6put6g76BPkAfqA/SB+mD9SH6UH2YPlwfoY/QR+qj9NH6GH2MPlZv1LcVxunj9Qn6RH2SPlmfrE/Rm/Rmfao+VZ+mT9dn6DP1WXqLHtCDelAP6dsJ2wlhPaJH9Fa9TW/XO/QOfbY+R5+rz9Pn6xvqG+kL9AX6Qn1j/WBmE30TfZG+qb6Zvrm+hb6lvljfSt9a30bfVt9O317fQd9R30nfWd9F31XfTd9d30PfU99LX6LvrUf1A4V9dEPv1GN6XDf1Lh2s8u3We/SEvlTv1ZN6Sk/rGT2rL9Nzel4v6EW9T1+u9+sr9H31/fT99QP0lfqB+kH6wfoh+qH6Yfrh+hH6kfpR+tH6Mfqx+nH68foq/YTSdaJ+kn6yfop+qn6afrp+hn6mfpN4k3iWfrZ+jn6ufp5+vn6BfqF+kX6xfol+qX6Zfrl+hX6lfpV+tX6Nvlpfo5/AXKtfp1+v36C/zt6o36TfrN+i36rfpt+u36Hfqd+l363fo9+r36ffrz+gP6g/pD+sP6I/qj+mP64/oT+pP6U/rT+jP6s/pz+vv6C/qL+kv6y/or+qv6a/rr+hv6mv1d/S39bf0d/V39Pf1z/QP9Q/0j/WP9E/1T/TP9e/0L/Uv9K/1r/Rv9W/09fp31v/ftB/tP79pP9s/ftF/9X695v+e+XfH/qf+l/639a/9TrRAP6RDZT1j25gGtgGroFvEBrEBqlBblAa1AatQW9oaNigYUDDwIZBDYMbhjQMbRjWMLxhRMPIhlENoxvGNIxtaGwY1zC+YULDxIZJDZMbpjQ0NTQ3TG2Y1jC9YUbDzIZZDevYloa1NPhv+Z/9q/a/YNUN+Busr25p8Ma6nwLydex9VPlvm11SSsGOtZa2KWcI+G8n18kFGoINoYZwQ6ShtaGtob2hvaGjYXbDnP+nL3gMzEIBwzBcw9e3Fp77zV5YttfWsrls1+Jqq+W1tmxj2bZtLJvLtr1q/Tm/z3XJT34qLX+VUayzrMqpvH47K6iiKqmy/jirqKqqqbr+OmuopmqptuqorgIU56yn+mqghmqkxmqipmqm5mqhlmql1mqjGNe2aqf26qCOquMS6NpJ/vEC1Vld1FXd1F095OkIUpB6ytujl4IVrBCFqLf6KFSh6qt+6q8BGqhBGqwwhWmIwhWuoYpQhCIVqWEarhEaqVEarTEaq3EarwmaqEmarCmaqmmarhmaqVmarShFaY7map58POZrgRZqkRZriZZqmZZrhVZqlVZrjdZqndZrgwI8NmqTNmuLtmqbMjm2a4d2apd2a4/2ap/264AO6pC8HId1REd1TMd1QicVrVPK7DitMzqrczqvC7qoS7qsK7qqa7quG7qpW7qtGN3RXd3TfT3QQz3SYz3R0/8+03O90Eu90mu90Vu903t90Ed90md90VdFu32Tr+O7fuinfinMI1a/9Ud/Fac4xSM+LrjiihvuuOMgAU48SIgniUhMEpIiwEhGclKQklSkJg1pSUd6MpCRTHiRmSxkJRveZCcHOclFbvKQl3zkpwAFKURhilCUYhSnBCXxoRS++FEaf8pQlnKUpwIVqURlqlCValSnBjWpRW3qUJcA6lGfBjSkEY1pQlOa0ZwWtKQVrWlDW9rRng50pBOBdKYLXelGd3oQRE96EUwIvelDKH3pR38GMJBBDCaMIYQzlAgiGcZwRjCSUYxmDGMZx3gmMJFJTGYKU5nGdGYwk1nMJoo5zGUe81nAQhaxmCUsZRnLWcFKVrGaNaxlHevZwEY2sZktbGUb29nBTnaxmz3sZR/7OcBBDnGYIxzlGMc5wUmiOcVpznCWc5znAhe5xGWucJVrXOcGN7nFbWK4w13ucZ8HPOQRj3nCU57xnBe85BWvecNb3vGeD3zkE5/5wle+8Z0f/OQXsfzmD3+JI57FNxdzNTdzN4clMKd5WELztESW2JJYUpNhZsksuaWwlJbKUlsaS2vpLL1lsIyWybwss2WxrJbNvC275bCclstyWx7La/ksvxWwglbIClsRK2rFrLiVsJLmY6XM1+L+h5/5WWnzt39TSZEYrroCAA==",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12074-dc94109";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true
    });
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
